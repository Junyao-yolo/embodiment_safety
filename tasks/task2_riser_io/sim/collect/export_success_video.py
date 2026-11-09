"""Export a successful recorded episode with synchronized three-camera views."""
from pathlib import Path
import argparse,json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from review_video import FFMPEG,FONT

def export(episode):
    episode=Path(episode)
    report=json.loads((episode/"result.json").read_text())
    if not report.get("accepted"):
        raise ValueError("A completed successful recording is required")
    diagnostics={k:report[k] for k in ("numerical_quality","diagnostic_checks","smoothness","command_limit_events","max_joint_gap_m","measured_thread_lead_m","thread_lead_fit_residual_m","largest_ik_residual","hand_tracking_errors_m","incomplete_reference_tracking_phases","measurements_path","mean_clamp_N") if k in report}
    diagnostics["role"]="Recorded numerical quality; not task acceptance or calibrated damage limits"
    (episode/"diagnostics.json").write_text(json.dumps(diagnostics,indent=2))
    meta=json.loads((episode/"episode.json").read_text())
    times=np.load(episode/"time.npy")
    names=("external","wrist_left","wrist_right")
    images={name:np.load(episode/(name+".npy"),mmap_mode="r") for name in names}
    out=Path("/mnt/data2/wjy_SERL/Embodied_Safety/outputs/task2_riser_io/review_success")/episode.name
    out.mkdir(parents=True,exist_ok=True)
    video=out/"success_three_views.mp4"
    width,height,tile=1152,488,384
    font=ImageFont.truetype(FONT,18)
    small=ImageFont.truetype(FONT,15)
    args=[FFMPEG,"-y","-hide_banner","-loglevel","warning","-f","rawvideo","-pix_fmt","rgb24",
          "-s",f"{width}x{height}","-r","10","-i","pipe:0","-an","-c:v","libx264",
          "-preset","veryfast","-crf","20","-threads","2","-pix_fmt","yuv420p","-movflags","+faststart",str(video)]
    with (out/"encoder.log").open("w") as log:
        pipe=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
        for i,t in enumerate(times):
            frame=Image.new("RGB",(width,height),(20,24,30))
            draw=ImageDraw.Draw(frame)
            draw.text((10,5),f"Task2 | {episode.name} | {t:.1f} s simulation | {meta['phases'][i]}",font=font,fill="white")
            for k,name in enumerate(names):
                label={"external":"EXTERNAL","wrist_left":"WRIST LEFT / hand -Y","wrist_right":"WRIST RIGHT / hand +Y"}[name]
                draw.text((k*tile+10,31),label,font=small,fill=(175,220,255))
                frame.paste(Image.fromarray(images[name][i]).resize((tile,tile)),(k*tile,55))
            draw.text((10,445),"Original 128x128 observations enlarged | 1x simulation time | All three views share the same timestamp",font=small,fill="white")
            draw.text((10,467),"Motion planning peak: 1 m/s; contact stages slowed | Safety forces are recorded, not damage thresholds",font=small,fill=(210,210,210))
            pipe.stdin.write(np.asarray(frame).tobytes())
        draw.rectangle((0,0,width,30),fill=(20,24,30))
        draw.text((10,5),"TASK COMPLETED | Terminal seated, screw installed, gripper released | Final frame held 3 s",font=font,fill=(110,240,160))
        for _ in range(30):pipe.stdin.write(np.asarray(frame).tobytes())
        pipe.stdin.close()
        if pipe.wait():raise RuntimeError("See "+str(out/"encoder.log"))
    (out/"video.json").write_text(json.dumps({"episode":str(episode),"video":str(video),"frames":len(times)+30,
        "fps":10,"duration_s":(len(times)+30)/10,"playback":"1x simulation time; final still 3 s",
        "source_resolution":[128,128],"accepted":True},indent=2)+"\n")
    print(video,flush=True)
    return video
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("episode",type=Path)
    export(parser.parse_args().episode)
