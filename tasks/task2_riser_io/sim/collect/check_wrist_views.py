"""Compare wrist camera mounts during a physical terminal grasp."""
from pathlib import Path
import json,sys
R=Path("/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io")
COLLECT=R/"sim"/"collect"
ARTIFACTS=Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io")
OUT=ARTIFACTS/"review_success/camera_opposed_v002"
OUT.mkdir(parents=True,exist_ok=True)
source=(R/"scripts/run_dp_bc.py").read_text()
source=source[:source.index(" phase='level';")]
source=source.replace("if out.exists():raise RuntimeError('episode already exists')","")
source=source.replace("ARTIFACTS/'logs'/'current_kit.log'","ARTIFACTS/'logs'/'camera_check_kit.log'")
capture=r'''
 def capture_camera_comparison(tag):
  from PIL import Image,ImageDraw
  setup_render()
  names=("wrist_left","wrist_right")
  old={n:UsdGeom.Xformable(stage.GetPrimAtPath(setup_render.paths[n])).GetLocalTransformation() for n in names}
  candidates={}
  for n,side in zip(names,(-1,1)):
   eye=Gf.Vec3d(side*.080,0,.035);aim=Gf.Vec3d(0,0,.110)
   candidates[n]=Gf.Matrix4d().SetLookAt(eye,aim,Gf.Vec3d(0,0,-1)).GetInverse()
  rows=[]
  for label,poses in (("OLD: same side mounts",old),("NEW: opposite side mounts",candidates)):
   for n,matrix in poses.items():
    UsdGeom.Xformable(stage.GetPrimAtPath(setup_render.paths[n])).MakeMatrixXform().Set(matrix)
   for _ in range(8):sim.render()
   images={n:setup_render.anns[n].get_data()[:,:,:3].copy() for n in ("external",)+names}
   for n,rgb in images.items():Image.fromarray(rgb).save(OUT/(tag+"_"+("old" if poses is old else "new")+"_"+n+".png"))
   rows.append((label,images))
  canvas=Image.new("RGB",(768,564),(20,24,30));draw=ImageDraw.Draw(canvas)
  for row,(label,images) in enumerate(rows):
   draw.text((6,row*282+4),tag+" | "+label+" | EXTERNAL / WRIST LEFT / WRIST RIGHT",fill="white")
   for col,n in enumerate(("external",)+names):
    canvas.paste(Image.fromarray(images[n]).resize((256,256)),(col*256,row*282+24))
  canvas.save(OUT/(tag+"_comparison.jpg"),quality=85)
  for n,matrix in old.items():UsdGeom.Xformable(stage.GetPrimAtPath(setup_render.paths[n])).MakeMatrixXform().Set(matrix)
  (OUT/"camera_poses.json").write_text(json.dumps({n:np.asarray(v).tolist() for n,v in candidates.items()},indent=2)+"\n")
  print("CAMERA_CHECK",tag,flush=True)
'''
source=source.replace(" phase='settle';tick(physics_hz*2)",capture+"\n phase='settle';tick(physics_hz*2)")
source=source.replace(" phase='lift';move("," capture_camera_comparison('grasp')\n phase='lift';move(")
source+=""" capture_camera_comparison('lift')
 (OUT/'result.json').write_text(json.dumps({'terminal_lift_observed':report['terminal_lift_observed'],'lift_height_m':report['lift_height_m'],'comparison':'actual grasp simulation; no physics steps between camera variants'},indent=2))
finally:
 if app is not None:app.close()
"""
compile(source,str(R/"scripts/run_dp_bc.py"),"exec")
sys.argv=[str(R/"scripts/run_dp_bc.py"),"--seed","101","--record"]
initial=json.loads((D/"data/raw/episode_0101/episode.json").read_text())["initial_conditions"]
exec(compile(source,str(R/"scripts/run_dp_bc.py"),"exec"),{"__name__":"__main__","OUT":OUT,"_collection_config":json.loads((COLLECT/"feedback_expert.json").read_text()),"_initial_conditions":initial})
