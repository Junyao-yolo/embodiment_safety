"""Compare best and final EMA policies on the complete held-out trajectory split."""
import argparse,collections,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from algos.diffusion_policy.common import ROOT,build
from algos.diffusion_policy.data import SequenceData
from algos.diffusion_policy.diagnostics import physical_metrics,label
from runners.eval_diffusion import write_json

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--training-run",required=True)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    run=Path(args.training_run).resolve();out=Path(args.output).resolve()
    if not out.is_relative_to(ROOT/"outputs"):raise ValueError("Outputs only")
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    ref=json.loads((run/"train/last.json").read_text())
    cfg=ref["config"]
    if ref["training_steps"]!=100000 or cfg["batch_size"]!=16:raise ValueError("Final training budget mismatch")
    data=SequenceData(cfg["data"],"validation",cfg["state_representation"],cfg["scope"])
    if data.meta.get('split_unit')!='episode' or len(data.meta['validation_episodes'])!=cfg['validation_episode_count']:
        raise ValueError('Validation episode count differs from training configuration')
    groups=collections.defaultdict(list);episodes=collections.defaultdict(list)
    for i,k in enumerate(data.starts):
        row=data.rows[k+1];groups[label(row)].append(i);episodes[row["episode"]].append(i)
    report=dict(status="running",scope="all held-out validation windows; no test trajectories",
        samples=len(data),episodes=list(episodes),seed=1234,batch_size=16,precision="float32",
        weights="ema",ddim_steps=cfg["num_inference_steps"],
        selection_rule="Minimum full-validation action_mse; exact tie chooses best.pt",
        candidates=[],network_updates=0)
    write_json(out/"validation_comparison.json",report)
    def device(b):
        return {k:device(v) for k,v in b.items()} if isinstance(b,dict) else b.cuda(non_blocking=True)
    reference_normalizer=None;reference_target=None
    for name in ("best","last"):
        path=run/"train"/(name+".pt")
        state=torch.load(path,map_location="cpu",weights_only=False)
        for key in ("data","state_representation","state_dim","scope","action_range_eps","prediction_type",
                    "dataset_manifest_sha256","rows_sha256","horizon","n_obs_steps","n_action_steps","num_inference_steps",
                    "down_dims","diffusion_step_embed_dim","rgb_features","image_size"):
            if state["config"][key]!=cfg[key]:raise ValueError("Checkpoint contract differs: "+key)
        norm={k:v for k,v in state["ema"].items() if k.startswith("normalizer.")}
        if reference_normalizer is None:reference_normalizer={k:v.clone() for k,v in norm.items()}
        elif set(norm)!=set(reference_normalizer) or not all(torch.equal(v,reference_normalizer[k]) for k,v in norm.items()):
            raise ValueError("Checkpoint normalization differs")
        model=build(state["config"]);model.load_state_dict(state["ema"]);model=model.cuda().eval()
        step=state["step"];del state
        loader=DataLoader(data,batch_size=16,shuffle=False,num_workers=2,pin_memory=True)
        predictions=[];targets=[];loss_sum=0.;count=0
        started=time.monotonic()
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(1234);torch.cuda.manual_seed_all(1234)
            for batch in loader:
                batch=device(batch);n=len(batch["action"])
                loss=model.compute_loss(batch)
                pred=model.predict_action(batch["obs"])["action"][:,0]
                target=batch["action"][:,1]
                if not torch.isfinite(loss) or not torch.isfinite(pred).all():raise ValueError("Nonfinite EMA validation")
                loss_sum+=float(loss)*n;count+=n
                predictions.append(pred.cpu().numpy());targets.append(target.cpu().numpy())
                if count%1600==0:
                    write_json(out/"progress.json",dict(candidate=name,completed_windows=count,total_windows=len(data),wall_s=time.monotonic()-started))
        pred=np.concatenate(predictions);target=np.concatenate(targets)
        if reference_target is None:reference_target=target.copy()
        elif not np.array_equal(target,reference_target):raise ValueError("Validation targets differ")
        metrics=physical_metrics(pred,target)
        row=dict(name=name,checkpoint=str(path),training_step=step,**metrics,diffusion_loss=loss_sum/count,
            per_axis_rmse=np.sqrt(np.mean((pred-target)**2,axis=0)).tolist(),
            per_stage={k:physical_metrics(pred[ids],target[ids]) for k,ids in groups.items()},
            per_episode={k:physical_metrics(pred[ids],target[ids]) for k,ids in episodes.items()},
            wall_s=time.monotonic()-started)
        np.savez_compressed(out/(name+"_predictions.npz"),window_start=np.asarray(data.starts),prediction=pred,target=target)
        report["candidates"].append(row);write_json(out/"validation_comparison.json",report)
        del model,loader;torch.cuda.empty_cache()
    chosen=min(report["candidates"],key=lambda x:x["action_mse"])
    report.update(status="finished",selected=chosen["name"],selected_checkpoint=chosen["checkpoint"],
                  normalization_equal=True,targets_equal=True,task_success_verified=False)
    write_json(out/"validation_comparison.json",report)
    text="# EMA validation comparison\n\n"+report["selection_rule"]+"\n\n"
    text+="|Checkpoint|Training step|Action MSE|Translation mm/s|Rotation deg/s|Gripper mm/s|\n|---|---:|---:|---:|---:|---:|\n"
    for x in report["candidates"]:
        text+=f"|{x['name']}|{x['training_step']}|{x['action_mse']:.8g}|{x['translation_rmse_mm_s']:.4f}|{x['rotation_rmse_deg_s']:.4f}|{x['gripper_rmse_mm_s']:.4f}|\n"
    text+="\nSelected: "+report["selected"]+". Full-task capability remains to be measured by the subsequent fixed-checkpoint simulation tests.\n"
    (out/"REPORT.md").write_text(text)
    print(json.dumps({k:v for k,v in report.items() if k not in ("candidates","episodes")}),flush=True)
if __name__=="__main__":main()
