"""Serve an official Task2 image policy to the DAgger collector."""
from pathlib import Path
import contextlib, io, json, struct, sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def main():
    with contextlib.redirect_stdout(sys.stderr):
        from learning.bootstrap import configure
        configure()
        import hydra, torch
        from learning.controller.task2_controller import ActionController
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        payload=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
        policy=hydra.utils.instantiate(payload["config"]["official_policy"])
        policy.load_state_dict(payload["model"],strict=True)
        execution=json.loads(sys.argv[3])
        policy.n_action_steps=int(execution["n_action_steps"])
        policy.to(sys.argv[2]).eval()
        model=ActionController(policy,fixed_noise=execution.get("fixed_noise",False))
        model.reset(execution.get("seed",42))
        del payload
    src=sys.stdin.buffer;dst=sys.stdout.buffer
    while True:
        head=src.read(4)
        if not head:break
        size=struct.unpack("!I",head)[0]
        with np.load(io.BytesIO(src.read(size)),allow_pickle=False) as data:
            if "clear_history" in data:
                model.history=[];model.pending=[];model.policy.reset()
                dst.write(b"R");dst.flush();continue
            if "clear_actions" in data:
                model.pending=[];dst.write(b"A");dst.flush();continue
            with contextlib.redirect_stdout(sys.stderr):
                commands=model.act({k:data[k] for k in ("external","wrist_left","wrist_right")},data["state"])
        dst.write(np.asarray(commands,dtype="<f4").tobytes());dst.flush()
if __name__=="__main__":main()
