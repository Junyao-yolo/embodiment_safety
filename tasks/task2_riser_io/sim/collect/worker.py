"""Task2 policy inference over a local binary pipe."""
import sys,struct,io,contextlib,json
import numpy as np
def main():
 with contextlib.redirect_stdout(sys.stderr):
  import torch
  from policy import Task2Policy
  torch.set_num_threads(2);torch.manual_seed(42)
  model=Task2Policy(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else "cpu",**(json.loads(sys.argv[3]) if len(sys.argv)>3 else {}))
 from interface import CAMERAS
 src=sys.stdin.buffer;dst=sys.stdout.buffer
 while True:
  head=src.read(4)
  if not head:break
  size=struct.unpack("!I",head)[0]
  with np.load(io.BytesIO(src.read(size)),allow_pickle=False) as data:
   if "clear_history" in data:
    model.clear_history();dst.write(b"R");dst.flush();continue
   if "clear_actions" in data:
    model.clear_actions();dst.write(b"A");dst.flush();continue
   x=data["state"];q=np.r_[x[:7],x[14]/2,x[14]/2];v=np.r_[x[7:14],0.,0.];u=np.r_[x[15:22],x[22]/2,x[22]/2]
   with contextlib.redirect_stdout(sys.stderr):
    commands=model.act({c:data[c] for c in CAMERAS},q,v,u)
  dst.write(np.asarray(commands,dtype="<f4").tobytes());dst.flush()
if __name__=="__main__":main()
