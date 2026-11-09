"""Execute a Task2 demonstration through the policy action decoder in Isaac Sim."""
from pathlib import Path
import argparse,json,sys
P=argparse.ArgumentParser();P.add_argument("--episode",required=True);a=P.parse_args()
episode=Path(a.episode).resolve();source_report=json.loads((episode/"result.json").read_text());seed=source_report["seed"]
R=Path(__file__).resolve().parents[2]
ARTIFACTS=Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io")
source=(R/"scripts/run_dp_bc.py").read_text()
source=source.replace("out=Path(globals().get('_episode_root',ARTIFACTS/'data'/'raw'))/('episode_'+str(args.seed).zfill(4))","out=ARTIFACTS/'replay'/('episode_'+str(args.seed).zfill(4))")
source=source.replace("ARTIFACTS/'logs'/'current_kit.log'","ARTIFACTS/'logs'/'replay_kit.log'")
# The physics/control loop is identical; replay needs no training images.
source=source.replace("if args.record and stepnum%(physics_hz//10)==0:write_frame(rec,tr,ring)","if False:write_frame(rec,tr,ring)")
start=source.index(" phase='settle';tick(physics_hz*2)")
end=source.index(" # Compare matching unloaded contact states",start)
replay="""
 sys.path.insert(0,str(R/'sim'/'collect'))
 from interface import action_block_to_targets
 replay_path=Path(REPLAY_EPISODE)
 replay_meta=json.loads((replay_path/'episode.json').read_text())
 replay_file='executed_action.npy' if (replay_path/'executed_action.npy').exists() else 'action.npy'
 replay_action=np.load(replay_path/replay_file,mmap_mode='r')
 report['replay_action_file']=replay_file
 replay_time=np.load(replay_path/'time.npy')
 phase='settle';tick(round(float(replay_time[0])*physics_hz))
 for k,block in enumerate(replay_action):
  phase=replay_meta['phases'][k]
  for command in action_block_to_targets(block):
   q=command.copy()
   robot.set_dof_position_targets(q[None],ix);tick(physics_hz//60)
 phase='full_verify';tick(physics_hz*3)
 hold=[r for r in records if r['phase']=='wire_verify']
 report['wire_seated']=bool(hold and ring_on_seat(hold[-1],target,assembly_geometry))
 report['replay_source']=str(replay_path)
 report['action_decoder']='interface.action_block_to_targets'
 report['thread_turns']=[]
"""
source=source[:start]+replay+source[end:]
start=source.index(" phase='screw_release';grip(.06,1.)")
end=source.index(" hold=[r for r in records if r['phase']=='full_verify']",start)
source=source[:start]+source[end:]
start=source.index(" if report.get('accepted',False) and args.record:")
end=source.index(" print('PERFORMANCE'",start)
source=source[:start]+""" out.mkdir(parents=True,exist_ok=True)
 report['performance']=performance_snapshot()
 (out/'result.json').write_text(json.dumps(report,indent=2))
"""+source[end:]
sys.argv=[str(R/"scripts/run_dp_bc.py"),"--seed",str(seed),"--record"]
exec(compile(source,str(R/"scripts/run_dp_bc.py"),"exec"),{"__name__":"__main__","_collection_config":dict(method="expert",randomize=True) if source_report.get("initial_conditions") else {}, "_initial_conditions":source_report.get("initial_conditions"),"_diagnostics_root":ARTIFACTS/"replay/attempt_results","REPLAY_EPISODE":str(episode)})
