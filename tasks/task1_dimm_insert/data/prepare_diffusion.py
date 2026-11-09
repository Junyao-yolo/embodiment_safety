"""Prepare accepted native 100 ms demonstrations with episode-held-out validation."""

import argparse
import json
import random
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from learning.dataset.task1_observation import CAMERAS, official_lowdim
from tasks.task1_dimm_insert.data.cache_rgb import build_cache
from tasks.task1_dimm_insert.env.policy import load_observation
from tasks.task1_dimm_insert.env.bc_dataset import command_at_observation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected", type=int, default=100)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--label", choices=("executed", "expert_clean"), default="executed"
    )
    args = parser.parse_args()
    source = Path(args.source).resolve()
    out = Path(args.output).resolve()
    source_manifest = json.loads((source / "manifest.json").read_text())
    entries = source_manifest["episodes"]
    n_eps = len(entries)
    if n_eps != args.expected or len({e["episode"] for e in entries}) != n_eps:
        raise ValueError(f"Expected {args.expected} distinct accepted episodes")
    for e in entries:
        if not all(e[k] for k in ("accepted", "full_task_success", "data_valid")):
            raise ValueError("Unaccepted source episode")
    order = list(range(n_eps))
    random.Random(args.seed).shuffle(order)
    n_valid = max(1, round(n_eps * args.validation_fraction))
    valid_eps = {entries[i]["episode"] for i in order[:n_valid]}
    counts = [
        json.loads((Path(e["output"]) / "result.json").read_text())["full_transitions"]
        for e in entries
    ]
    total = sum(counts)
    out.mkdir(parents=True, exist_ok=False)
    arrays = {
        "state": np.lib.format.open_memmap(
            out / "state.npy", mode="w+", dtype=np.float32, shape=(total, 31)
        ),
        "proprio9": np.lib.format.open_memmap(
            out / "proprio9.npy", mode="w+", dtype=np.float32, shape=(total, 9)
        ),
        "action": np.lib.format.open_memmap(
            out / "action.npy", mode="w+", dtype=np.float32, shape=(total, 7)
        ),
    }
    images = {
        k: np.lib.format.open_memmap(
            out / (k + ".npy"), mode="w+", dtype=np.uint8, shape=(total, 256, 256, 3)
        )
        for k in CAMERAS
    }
    split_lists = {"train": [], "validation": []}
    rows = []
    episodes = []
    offset = 0
    start = time.monotonic()

    def decode(item):
        row, episode, progress = item
        return load_observation(row["observations"], episode, progress, max_age_s=0.1)

    with ThreadPoolExecutor(max_workers=6) as pool:
        for ep_index, (entry, expected) in enumerate(zip(entries, counts)):
            episode = Path(entry["episode"])
            parent = Path(entry["output"])
            transitions = [
                json.loads(line) for line in (episode / "transitions.jsonl").open()
            ]
            if len(transitions) != expected:
                raise ValueError("Transition count mismatch: " + str(episode))
            first = transitions[0]["observations"]["sim_time"]
            previous = None
            items = []
            local_rows = []
            for tick, row in enumerate(transitions):
                o = row["observations"]
                nxt = row["next_observations"]
                if (
                    row["index"] != tick
                    or abs(o["sim_time"] - first - tick * 0.1) > 1e-4
                ):
                    raise ValueError("Noncontiguous observation boundary")
                if previous is not None and o != previous:
                    raise ValueError(
                        "Next observation does not equal next transition observation"
                    )
                previous = nxt
                action, duration, diag = command_at_observation(row)
                if (
                    abs(duration - 0.1) > 1e-6
                    or diag["physics_steps"] != 24
                    or len(diag["controller_commands"]) != 24
                ):
                    raise ValueError("Not a native 100 ms transition")
                if args.label == "expert_clean":
                    action = np.asarray(
                        row["diagnostics"]["expert_clean_action"], dtype=np.float32
                    )
                    if (
                        action.shape != (7,)
                        or not np.isfinite(action).all()
                        or np.abs(action).max() > 1.00001
                    ):
                        raise ValueError("Invalid expert_clean_action label")
                label = json.loads(
                    (parent / f"stage_transition_{tick:03d}.json").read_text()
                )
                if label["index"] != tick:
                    raise ValueError("Stage-history index mismatch")
                arrays["action"][offset + tick] = action
                arrays["proprio9"][offset + tick] = official_lowdim(o)
                items.append((row, episode, label["progress_before"]))
                local_rows.append(
                    dict(
                        episode=str(episode),
                        tick=tick,
                        source_index=tick,
                        stage=diag["expert_stage"],
                        source_kind="native_collection",
                    )
                )
            for tick, obs in enumerate(pool.map(decode, items)):
                arrays["state"][offset + tick] = obs["state"][0]
                for k in CAMERAS:
                    images[k][offset + tick] = obs[k][0]
            split = "validation" if str(episode) in valid_eps else "train"
            starts = list(range(offset, offset + expected - 16 + 1))
            split_lists[split].extend(starts)
            rows.extend(local_rows)
            episodes.append(
                dict(
                    episode=str(episode),
                    split=split,
                    rows=expected,
                    sequences=len(starts),
                    offset=offset,
                    seed=entry["seed"],
                )
            )
            offset += expected
            for array in [*arrays.values(), *images.values()]:
                array.flush()
            progress = dict(
                episodes_done=ep_index + 1,
                episodes_total=n_eps,
                rows_done=offset,
                rows_total=total,
                elapsed_s=time.monotonic() - start,
            )
            (out / "prepare_status.json").write_text(json.dumps(progress, indent=2))
            print(json.dumps(progress), flush=True)
    train_eps = [e["episode"] for e in episodes if e["split"] == "train"]
    val_eps = [e["episode"] for e in episodes if e["split"] == "validation"]
    if set(train_eps) & set(val_eps):
        raise ValueError("Episode split overlap")
    meta = dict(
        kind="native_collection_dp_sequences_v001",
        horizon=16,
        n_obs_steps=2,
        **split_lists,
        rows=total,
        episodes=episodes,
        source=str(source),
        split_unit="episode",
        split_seed=args.seed,
        train_episodes=train_eps,
        validation_episodes=val_eps,
        independent_demonstrations=n_eps,
        action_label=args.label,
        validation_scope="held-out whole trajectories from the same collection distribution",
        source_kind="native_collection",
        control_hz=10,
        physics_hz=240,
        controller_solver=source_manifest.get("controller", "dogbox_trf"),
        controllers=source_manifest.get("controllers", []),
        action_interface="normalized_world_velocity_100ms",
        notes=[
            "Only accepted native trajectories; actions held for 24 physics steps.",
            "State31 contains four causal progress_before bits; state33 conversion uses the shared adapter.",
            "No sequence crosses an episode boundary. Normalization is fitted on training episodes only.",
            "RGB decoded and checked during preparation; original 256x256 pixels retained.",
        ],
    )
    (out / "rows.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (out / "manifest.json").write_text(json.dumps(meta, indent=2))
    build_cache(out, out / "lossless_rgb_cache")
    (out / "prepare_status.json").write_text(
        json.dumps(
            dict(
                status="finished",
                rows=total,
                train_episodes=len(train_eps),
                validation_episodes=len(val_eps),
                train_sequences=len(split_lists["train"]),
                validation_sequences=len(split_lists["validation"]),
                elapsed_s=time.monotonic() - start,
            ),
            indent=2,
        )
    )
    print((out / "prepare_status.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
