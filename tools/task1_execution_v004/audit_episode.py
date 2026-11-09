"""Audit saved 100 ms transfers and recorded contact measurements."""
import sys,json,math
from pathlib import Path
def audit(path,source):
    episodes=list(path.glob("episodes/*/transitions.jsonl"))
    if len(episodes)!=1:raise ValueError("Expected one episode")
    episode=episodes[0].parent
    maximum_penetration=0.;responsible=None;force_peaks={};physics_count=0
    pair_peaks={}
    with (episode/"physics.jsonl").open() as f:
        for line in f:
            x=json.loads(line);physics_count+=1
            for k,v in x["metrics"].items():
                if k.endswith("_N"):force_peaks[k]=max(force_peaks.get(k,0.),float(v))
            for c in x["contacts"]:
                if "/World/MatchedDIMM/" not in c["a"]+c["b"]:continue
                depth=max(0.,-c["separation_m"])
                if depth>maximum_penetration:
                    maximum_penetration=depth
                    responsible=dict(time=x["sim_time"],a=c["a"],b=c["b"],normal_force_N=c["normal_force_N"],point=c["point"])
                key=c["a"]+" <-> "+c["b"]
                pair_peaks[key]=max(pair_peaks.get(key,0.),depth)
    source_rows=[json.loads(s) for s in Path(source).read_text().splitlines()]
    n=0;maxdt=0.;maxaction=0.;images=set();previous=None;bad=[]
    with episodes[0].open() as f:
        for line in f:
            x=json.loads(line);i=x["index"];n+=1
            o=x["observations"];nxt=x["next_observations"]
            maxdt=max(maxdt,abs(nxt["sim_time"]-o["sim_time"]-.1))
            if previous is not None and abs(previous-o["sim_time"])>1e-6:bad.append("transition_gap")
            previous=nxt["sim_time"]
            if len(x["diagnostics"]["controller_commands"])!=24:bad.append("physics_commands")
            maxaction=max(maxaction,max(abs(a-b) for a,b in zip(x["actions"],source_rows[i]["actions"])))
            for y in [o,nxt]:
                for key,info in y["images"].items():
                    name=info["file"];images.add(name)
                    if not (episode/name).is_file():bad.append("missing_image:"+name)
                    if not info["quality"]["valid"]:bad.append("invalid_image")
    result=dict(transitions=n,physics_records=physics_count,image_files=len(images),
        max_dt_error_s=maxdt,max_action_difference=maxaction,alignment_errors=bad,
        maximum_recorded_dimm_penetration_m=maximum_penetration,responsible_contact=responsible,
        force_peaks_N=force_peaks,penetration_pairs=sorted(pair_peaks.items(),key=lambda x:x[1],reverse=True)[:10],
        transfer_alignment_passed=not bad and maxdt<=1e-6 and maxaction==0,
        contact_scope="Measured PhysX contacts only; not an exhaustive solid-intersection proof.")
    return result
if __name__=="__main__":
    source=json.loads((Path(__file__).parent/"contract.json").read_text())["source"]
    result=audit(Path(sys.argv[1]),source)
    Path(sys.argv[2]).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
