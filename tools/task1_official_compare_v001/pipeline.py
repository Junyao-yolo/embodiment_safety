"""Sequential GPU training and matched Task1 evaluation."""
import common
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from evaluate import wait_gpu


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True,type=Path);args=ap.parse_args()
    run=args.run.resolve();plan=json.loads((run/'plan.json').read_text())
    status=dict(pid=os.getpid(),started_at=time.time(),status='starting',completed=[])
    def phase(name,command):
        status.update(status='waiting_for_gpu',phase=name);common.write_json(run/'status.json',status)
        wait_gpu()
        status.update(status='running',phase=name);common.write_json(run/'status.json',status)
        with (run/(name+'.log')).open('a') as log:
            proc=subprocess.Popen(command,cwd=common.ROOT,stdout=log,stderr=subprocess.STDOUT)
            status['child_pid']=proc.pid;common.write_json(run/'status.json',status)
            code=proc.wait()
        if code:
            status.update(status='failed',returncode=code);common.write_json(run/'status.json',status)
            raise RuntimeError(f'{name} failed: {code}')
        status['completed'].append(name);common.write_json(run/'status.json',status)
    for path in plan['configs']:
        cfg=json.loads(Path(path).read_text())
        phase(cfg['input_variant']+'_timing700',[sys.executable,'-u',str(common.HERE/'train.py'),'--config',path,
            '--stop-after','700'])
    estimates={}
    for path in plan['configs']:
        cfg=json.loads(Path(path).read_text());out=Path(cfg['output_dir'])
        timing=json.loads((out/'timing.json').read_text())
        records=[json.loads(line) for line in (out/'metrics.jsonl').open()]
        validation=[x['seconds'] for x in records if x.get('event')=='validation']
        saves=[x['seconds'] for x in records if x.get('event')=='checkpoint']
        update_s=timing['means']['update_ms']/1000
        val_s=sum(validation)/len(validation);save_s=sum(saves)/len(saves)
        stage=cfg['stage_updates']
        estimates[cfg['input_variant']]=dict(stable_update_ms=update_s*1000,
            stage_updates=stage,stage_training_hours=stage*update_s/3600,
            full_3000_epoch_training_hours=cfg['updates']*update_s/3600,
            validation_seconds=val_s,checkpoint_seconds=save_s,
            stage_validation_and_save_hours=(stage/cfg['validate_every']*val_s+
                (stage/cfg['checkpoint_every']+stage/cfg['validate_every'])*save_s)/3600,
            excludes_evaluation=True)
    common.write_json(run/'duration_estimate.json',estimates)
    for path in plan['configs']:
        cfg=json.loads(Path(path).read_text())
        phase(cfg['input_variant']+'_train',[sys.executable,'-u',str(common.HERE/'train.py'),'--config',path,
            '--stop-after',str(cfg['stage_updates'])])
    for path in plan['configs']:
        cfg=json.loads(Path(path).read_text());out=Path(cfg['output_dir'])
        phase(cfg['input_variant']+'_eval20',[sys.executable,'-u',str(common.HERE/'evaluate.py'),
            '--checkpoint',str(out/'checkpoints'/f"step_{cfg['stage_updates']:07d}.pt"),
            '--output',str(out/'eval20'),'--episodes','20'])
    comparison={}
    for path in plan['configs']:
        cfg=json.loads(Path(path).read_text());out=Path(cfg['output_dir'])
        comparison[cfg['input_variant']]=dict(config=cfg,
            training=json.loads((out/f"segment_{cfg['stage_updates']:07d}.json").read_text()),
            evaluation=json.loads((out/'eval20/summary.json').read_text()),
            timing=json.loads((out/'timing.json').read_text()))
    common.write_json(run/'comparison.json',comparison)
    status.update(status='stage_comparison_complete',finished_at=time.time());common.write_json(run/'status.json',status)


if __name__=='__main__':main()
