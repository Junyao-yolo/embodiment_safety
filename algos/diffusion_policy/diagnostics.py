"""Source/time-aware offline metrics. Fit probes are never held-out success tests."""
from collections import defaultdict
import numpy as np

def label(row):
    if row.get("source_kind") == "native_collection":
        return "native/" + row["stage"]
    if row["stage"] == "native":
        start = (int(row["tick"]) // 50) * 5
        return "native_time_%03d_%03ds" % (start, start+5)
    return "historical/" + row["stage"]

def balanced_ids(data, per_group=8, per_episode=False):
    groups = defaultdict(list)
    for i, k in enumerate(data.starts):
        row=data.rows[k+1]
        key=(row["episode"],label(row)) if per_episode else label(row)
        groups[key].append(i)
    return sorted(i for ids in groups.values()
                  for i in np.asarray(ids)[np.linspace(0,len(ids)-1,min(per_group,len(ids)),dtype=int)].tolist())

def coverage(data, ids):
    groups = defaultdict(int)
    for i in ids:
        groups[label(data.rows[data.starts[i]+1])] += 1
    return dict(groups)

def physical_metrics(pred, target):
    err = pred-target
    target_v = target[:,:3]*320.
    pred_v = pred[:,:3]*320.
    mask = np.linalg.norm(target_v,axis=1)>1.
    opposite = (pred_v*target_v).sum(axis=1)<0
    down = target_v[:,2]<-0.5
    return dict(samples=len(pred), action_mse=float(np.mean(err**2)),
                translation_rmse_mm_s=float(np.sqrt(np.mean((err[:,:3]*320.)**2))),
                rotation_rmse_deg_s=float(np.sqrt(np.mean((err[:,3:6]*90.)**2))),
                gripper_rmse_mm_s=float(np.sqrt(np.mean((err[:,6]*60.)**2))),
                opposite_translation_count=int((opposite & mask).sum()),
                nontrivial_translation_count=int(mask.sum()),
                downward_target_count=int(down.sum()),
                wrong_upward_over_1mm_s_count=int((down & (pred_v[:,2]>1.)).sum()))
