"""
ingest.py -- turn any canonical-format video+CSV pair into a registered clip.

This is the one place any new clip -- real, synthetic (synthesize.py), or
augmented (augment.py) -- becomes images/+labels/ and enters the registry.
The box-drawing logic is copied from the reference data_process.py, which
Phase 1 confirmed is already fully source-agnostic (it only ever reads
frame/x/y, never anything real-video-specific).

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.generate.ingest --video out/video.mp4 --csv out/trajectories.csv \\
        --tier 40 --source synthetic --split train
"""
import argparse
import shutil
from datetime import datetime
from pathlib import Path

import cv2

from data_engine.common.trajectory_io import load_tracks
from data_engine.registry.registry import REPO_ROOT, Registry, bin_real_density, measure_avg_density
from data_engine.registry.schema import Clip

BOX_SIZE = 30
HALF_BOX = BOX_SIZE // 2
CLASS_ID = 0

DATA_ENGINE_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def ingest_clip(video_path, csv_path, out_dir) -> int:
    """
    Draw fixed-size YOLO boxes at every (frame, x, y) in csv_path onto the
    frames of video_path, writing out_dir/images/, out_dir/labels/, and
    out_dir/images_wbox/ (visual QA copies with green boxes drawn).

    Returns the number of frames processed.
    """
    out_dir = Path(out_dir)
    dir_img = out_dir / "images"
    dir_lbl = out_dir / "labels"
    dir_vis = out_dir / "images_wbox"
    for d in (dir_img, dir_lbl, dir_vis):
        d.mkdir(parents=True, exist_ok=True)

    df = load_tracks(csv_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current = df[df["frame"] == frame_idx]
        yolo_lines = []
        vis_frame = frame.copy()

        for _, row in current.iterrows():
            cx, cy = float(row["x"]), float(row["y"])

            top_left = (int(cx - HALF_BOX), int(cy - HALF_BOX))
            bottom_right = (int(cx + HALF_BOX), int(cy + HALF_BOX))
            cv2.rectangle(vis_frame, top_left, bottom_right, (0, 255, 0), 2)

            norm_x = _clamp01(cx / width)
            norm_y = _clamp01(cy / height)
            norm_w = _clamp01(BOX_SIZE / width)
            norm_h = _clamp01(BOX_SIZE / height)
            yolo_lines.append(f"{CLASS_ID} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}")

        filename_base = f"frame_{frame_idx:06d}"
        cv2.imwrite(str(dir_vis / f"{filename_base}.jpg"), vis_frame)
        cv2.imwrite(str(dir_img / f"{filename_base}.jpg"), frame)
        with open(dir_lbl / f"{filename_base}.txt", "w") as f:
            f.write("\n".join(yolo_lines))

        frame_idx += 1

    cap.release()
    return frame_idx


def ingest(video_path, csv_path, tier=None, source_type="real", split="train", clip_id=None,
           out_dir=None, label_status="human") -> Clip:
    """
    label_status defaults to "human" for all source types here: real clips
    are human-labeled, and synthetic/augmented ground truth is exact by
    construction -- neither is a model prediction awaiting correction.
    "model_bootstrap"/"model_bootstrap+corrected" are for the future
    bootstrap-labeling stage, which registers clips explicitly as such.

    `tier` -- for synthetic, this is the generation target (Simulation's
    num_agents) and must be supplied explicitly; it isn't something the
    resulting CSV alone can tell you (an agent crossing the frame boundary
    spawns a new ID, so the actual output can drift from the target). For
    real AND augmented clips it's optional: unlike synthetic, their density
    is exactly what the resulting CSV measures (augmented is just real clips
    combined -- no separate "target" concept), so if omitted it's
    auto-computed via registry.bin_real_density() from the CSV's measured
    avg sperm/frame, which is the recommended default -- pass it explicitly
    only to override that measurement.
    """
    measured_density = None
    if tier is None:
        if source_type not in ("real", "augmented"):
            raise ValueError(f"--tier is required for source_type={source_type!r} "
                              f"(it's the generation target, not something the CSV alone determines)")
        measured_density = round(measure_avg_density(csv_path), 1)
        tier = bin_real_density(measured_density)
    elif source_type in ("real", "augmented"):
        # Still record the measurement even when tier is given explicitly,
        # so it's not lost for future re-binning.
        measured_density = round(measure_avg_density(csv_path), 1)

    if clip_id is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        clip_id = f"{source_type}_{tier}_{stamp}"

    if out_dir is None:
        out_dir = DATA_ENGINE_DATA_DIR / clip_id
    out_dir = Path(out_dir)

    n_frames = ingest_clip(video_path, csv_path, out_dir)

    video_dest = out_dir / f"{clip_id}.mp4"
    csv_dest = out_dir / f"{clip_id}_trackpy_format.csv"
    shutil.copy(video_path, video_dest)
    shutil.copy(csv_path, csv_dest)

    clip = Clip(
        clip_id=clip_id,
        path=str(out_dir.relative_to(REPO_ROOT)).replace("\\", "/") if out_dir.is_relative_to(REPO_ROOT)
             else str(out_dir),
        source_type=source_type,
        density_tier=tier,
        split=split,
        label_status=label_status,
        notes=f"ingested from {Path(video_path).name} + {Path(csv_path).name}",
        source_video=str(video_dest.relative_to(REPO_ROOT)).replace("\\", "/")
                     if video_dest.is_relative_to(REPO_ROOT) else str(video_dest),
        ground_truth_csv=str(csv_dest.relative_to(REPO_ROOT)).replace("\\", "/")
                         if csv_dest.is_relative_to(REPO_ROOT) else str(csv_dest),
        measured_density=measured_density,
    )

    registry = Registry.load()
    registry.add_clip(clip)
    registry.save()

    print(f"Ingested {n_frames} frames -> {out_dir}")
    print(f"Registered clip_id={clip_id!r} (tier={tier}, source={source_type}, split={split})")
    return clip


def _cli():
    parser = argparse.ArgumentParser(description="Ingest a video+CSV pair as a registered clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--tier", default=None,
                         help="Synthetic: required -- the generation num_agents (any positive "
                              "integer), e.g. 40. Real/augmented: optional -- auto-computed from "
                              "the CSV's measured density if omitted (recommended).")
    parser.add_argument("--source", required=True, choices=["real", "synthetic", "augmented"])
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--clip-id", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--label-status", default="human",
                         choices=["human", "model_bootstrap", "model_bootstrap+corrected"])
    args = parser.parse_args()

    ingest(args.video, args.csv, args.tier, args.source, args.split,
           clip_id=args.clip_id, out_dir=args.out_dir, label_status=args.label_status)


if __name__ == "__main__":
    _cli()
