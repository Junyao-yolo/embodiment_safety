from pathlib import Path
import numpy as np,trimesh as tm
from pxr import Usd,UsdGeom,UsdPhysics,Sdf,Gf
R=Path('/mnt/data2/wjy_SERL/Embodied_Safety/tasks/task2_riser_io')
dest=R/'scene/riser_io_success_v014.usda'
if dest.exists():dest.unlink()
s=Usd.Stage.CreateNew(str(dest));s.GetRootLayer().subLayerPaths=['riser_io_screw_access_v012.usda']
cache=UsdGeom.XformCache()
def mesh_world(path):
 p=UsdGeom.Mesh(s.GetPrimAtPath(path));m=cache.GetLocalToWorldTransform(p.GetPrim())
 v=np.array([m.Transform(Gf.Vec3d(*map(float,v))) for v in p.GetPointsAttr().Get()])
 return tm.Trimesh(v,np.array(p.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),process=True)
def write(path,g):
 assert g.is_watertight and g.is_volume,path
 p=UsdGeom.Mesh(s.GetPrimAtPath(path));inv=cache.GetLocalToWorldTransform(p.GetPrim()).GetInverse()
 v=np.array([inv.Transform(Gf.Vec3d(*map(float,x))) for x in g.vertices])
 p.GetPointsAttr().Set([Gf.Vec3f(*map(float,x)) for x in v])
 p.GetFaceVertexCountsAttr().Set([3]*len(g.faces));p.GetFaceVertexIndicesAttr().Set(g.faces.reshape(-1).tolist())
 p.GetExtentAttr().Set([Gf.Vec3f(*v.min(0)),Gf.Vec3f(*v.max(0))])
def cylinder(r,h,c):
 g=tm.creation.cylinder(radius=r,height=h,sections=96);g.apply_translation(c);return g
tx,ty,tz=-.092,-.071,.818
# Annular seat supports the ring disk while leaving the barrel grasp accessible.
old=mesh_world('/PCB_4_Terminal/SingleSeat')
round_envelope=cylinder(.0134,.012,[tx,ty,tz-.002])
seat=tm.boolean.intersection([old,round_envelope],engine='manifold')
# Lead-in pocket accommodates millimetric placement error without changing the terminal.
pocket=tm.creation.revolve(np.array([[0.,.818],[.0123,.818],[.0132,.8195],[.0132,.825],[0.,.825]]),sections=96)
pocket.apply_translation([tx,ty,0.])
seat=tm.boolean.difference([seat,pocket],engine='manifold')
write('/PCB_4_Terminal/SingleSeat',seat)
# A rear relief in the base clears both fingers; side walls retain the nut support.
base=mesh_world('/PCB_holder/SingleBase')
slot=tm.creation.box(extents=[.030,.042,.030]);slot.apply_translation([tx,ty+.030,.808])
write('/PCB_holder/SingleBase',tm.boolean.difference([base,slot],engine='manifold'))
support=mesh_world('/PCB_holder/LoadSupportEnvelope')
write('/PCB_holder/LoadSupportEnvelope',tm.boolean.difference([support,slot],engine='manifold'))
# Narrow saddle under the wire leaves the two sides accessible to the fingers.
landing=mesh_world('/World/TargetCableLanding')
landing.vertices[:,0]=tx+(landing.vertices[:,0]-tx)*(.006/.024)
# Clearance follows the downward cable exit tangent; the saddle must not lift the ring.
upper=landing.vertices[:,2]>.8001
blend=np.clip((.049-landing.vertices[:,1])/.062,0.,1.)
landing.vertices[upper,2]-=.002*blend[upper]
write('/World/TargetCableLanding',landing)
for i in range(23):
 p=s.GetPrimAtPath(f'/wire1/Bend_{i:02d}')
 lim=UsdPhysics.LimitAPI.Apply(p,'rotX');lim.CreateLowAttr(-80.);lim.CreateHighAttr(80.)
 d=UsdPhysics.DriveAPI.Apply(p,'rotX');d.CreateTypeAttr('force');d.CreateStiffnessAttr(.0001);d.CreateDampingAttr(.00002);d.CreateMaxForceAttr(.05)
ring='/wire1/_0347300_ring_terminal/Mesh'
original=Usd.Stage.Open(str(R/'scene/riser_io_screw_access_v012.usda'))
for attr in ['points','faceVertexIndices','faceVertexCounts']:
 assert s.GetPrimAtPath(ring).GetAttribute(attr).Get()==original.GetPrimAtPath(ring).GetAttribute(attr).Get()
s.GetRootLayer().customLayerData={'ringGeometryUnchanged':True,'calibrationStatus':'candidate finite torsion; material calibration pending','purpose':'finger clearance and complete assembly validation'}
x=UsdGeom.Xformable(s.GetPrimAtPath('/World/Robot'));x.ClearXformOpOrder();x.AddTranslateOp().Set(Gf.Vec3d(-.52,-.27,.75))
s.GetRootLayer().Save()
print(dest)
