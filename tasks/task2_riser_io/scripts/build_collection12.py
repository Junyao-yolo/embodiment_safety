"""Create a twelve-body cable with the original twenty-four collision capsules."""
from pathlib import Path
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Gf
import numpy as np
from scipy.spatial.transform import Rotation
R=Path(__file__).resolve().parents[1]
source=R/"scene/riser_io_success_v014.usda"
target=R/"scene/riser_io_collection12_v001.usda"
base=Usd.Stage.Open(str(source));flat=base.Flatten()
world={str(p.GetPath()):UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) for p in base.Traverse() if str(p.GetPath()).startswith("/wire1/Segment_")}
stage=Usd.Stage.CreateNew(str(target));stage.GetRootLayer().subLayerPaths=[source.name]
cache=UsdGeom.XformCache()
odd_to_even={f"/wire1/Segment_{i:02d}":f"/wire1/Segment_{i-1:02d}" for i in range(1,24,2)}
for odd,even in odd_to_even.items():
 child=even+"/SecondCapsule"
 Sdf.CreatePrimInLayer(stage.GetRootLayer(),even)
 Sdf.CopySpec(flat,odd,stage.GetRootLayer(),child)
 p=stage.GetPrimAtPath(child)
 p.RemoveAPI(UsdPhysics.RigidBodyAPI);p.RemoveAPI(UsdPhysics.MassAPI)
 p.RemoveAPI(UsdPhysics.FilteredPairsAPI)
 xf=UsdGeom.Xformable(p);xf.ClearXformOpOrder();xf.AddTransformOp().Set(world[odd]*world[even].GetInverse())
 mass0=UsdPhysics.MassAPI(base.GetPrimAtPath(even)).GetMassAttr().Get()
 mass1=UsdPhysics.MassAPI(base.GetPrimAtPath(odd)).GetMassAttr().Get()
 UsdPhysics.MassAPI.Apply(stage.GetPrimAtPath(even)).CreateMassAttr(mass0+mass1)
 stage.GetPrimAtPath(odd).SetActive(False)
for p in list(stage.Traverse()):
 if not p.IsA(UsdPhysics.Joint):continue
 j=UsdPhysics.Joint(p);bodies=[j.GetBody0Rel().GetTargets(),j.GetBody1Rel().GetTargets()]
 mapped=[odd_to_even.get(str(x[0]),str(x[0])) if x else "" for x in bodies]
 if mapped[0] and mapped[0]==mapped[1]:
  p.SetActive(False);continue
 for side,targets in enumerate(bodies):
  if not targets or str(targets[0]) not in odd_to_even:continue
  old=str(targets[0]);new=odd_to_even[old];rel=world[old]*world[new].GetInverse()
  pos=p.GetAttribute(f"physics:localPos{side}");rot=p.GetAttribute(f"physics:localRot{side}")
  xyz=rel.Transform(Gf.Vec3d(pos.Get()))
  q=rot.Get();q0=Rotation.from_quat([*q.GetImaginary(),q.GetReal()])
  qr=Rotation.from_matrix(np.asarray(rel)[:3,:3].T)*q0;qq=qr.as_quat()
  pos.Set(Gf.Vec3f(*xyz));rot.Set(Gf.Quatf(float(qq[3]),Gf.Vec3f(*qq[:3])))
  p.GetRelationship(f"physics:body{side}").SetTargets([new])
 if str(p.GetPath()).startswith("/wire1/Bend_"):
  for axis in ["rotX","rotY","rotZ"]:
   drive=UsdPhysics.DriveAPI(p,axis)
   drive.GetStiffnessAttr().Set(drive.GetStiffnessAttr().Get()/2)
   drive.GetDampingAttr().Set(drive.GetDampingAttr().Get()/2)
# Verify initial collider poses and terminal geometry are preserved.
for odd,even in odd_to_even.items():
 actual=UsdGeom.Xformable(stage.GetPrimAtPath(even+"/SecondCapsule")).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
 if np.max(np.abs(np.asarray(actual)-np.asarray(world[odd])))>1e-8:raise RuntimeError("Initial capsule placement changed")
ring="/wire1/_0347300_ring_terminal"
for attr in ["points","faceVertexCounts","faceVertexIndices"]:
 a=base.GetPrimAtPath(ring).GetAttribute(attr).Get();b=stage.GetPrimAtPath(ring).GetAttribute(attr).Get()
 if not np.array_equal(a,b):raise RuntimeError("Terminal geometry changed")
stage.GetRootLayer().Save()
print(target)
