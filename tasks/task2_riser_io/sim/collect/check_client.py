"""Exercise the actual policy pipe and flush cached actions across a wait."""
from pathlib import Path
import numpy as np
from client import PolicyClient
from interface import CAMERAS
D=Path(__file__).resolve().parent
p=D/"data/raw/episode_0020"
state=np.load(p/"state.npy")[0]
images={c:np.load(p/(c+".npy"),mmap_mode="r")[0] for c in CAMERAS}
client=PolicyClient(D/"runs/pilot5/policy.pt",n_action_steps=8,fixed_noise=True,seed=103)
try:
 first=client.act(images,state)
 client.act(images,state)
 client.clear_history()
 again=client.act(images,state)
 assert first.shape==(6,9)
 np.testing.assert_array_equal(first,again)
 print("Policy pipe: 6x9 commands and explicit post-wait replan passed.")
finally:client.close()
