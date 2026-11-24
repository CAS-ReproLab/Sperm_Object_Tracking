"""
bootstrap_label.py -- the piece that actually closes the data engine loop.

Runs an initial tracker on an unlabeled video, converts its output into the
exact CSV format the labeler GUI already reads/writes, then launches the
labeler pre-loaded with those predictions so a human corrects them instead
of labeling the video from nothing. Once the labeler window closes, if the
human actually saved corrections, ingests the result as a new registered
clip with label_status="model_bootstrap+corrected".

Two tracker backends, selected by --backend (default "auto"):
  - yolo:     a trained model + ByteTrack/BoT-SORT (stages/validate/track.py).
              Needs --model. Requires trajectory_io.from_bytetrack_output()
              to get to the canonical frame,sperm,x,y shape.
  - trackpy:  UnifiedFramework/tracker.py's centroid detection + linking,
              vendored in trackpy_tracker.py -- what the original videos
              were actually started with, before any YOLO model existed.
              Already produces frame,sperm,x,y directly, no conversion step.
"auto" picks yolo if --model is given, else trackpy -- so before any model
exists, this falls back to trackpy automatically, and trackpy stays
available as an explicit choice (--backend trackpy) even once a model
exists, rather than being a separate stage of its own.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.label.bootstrap_label --model training_weights/finetune_1/weights/best.pt \\
        --video path/to/new_video.mp4 --split train

    # no model yet -- falls back to trackpy automatically
    python -m data_engine.stages.label.bootstrap_label --video path/to/new_video.mp4 --split train
"""
import argparse
from pathlib import Path

from data_engine.common.trajectory_io import from_bytetrack_output, save_tracks
from data_engine.stages.generate.ingest import ingest
from data_engine.stages.label.correction_flow import launch_labeler_and_wait
from data_engine.stages.label.trackpy_tracker import (
    DEFAULT_DIAMETER, DEFAULT_MAXSIZE, DEFAULT_MEMORY, DEFAULT_MINMASS,
    DEFAULT_SEARCH_RANGE, run_trackpy_tracking,
)
from data_engine.stages.validate.track import run_tracking


def _resolve_backend(backend, model_path):
    if backend == "auto":
        return "yolo" if model_path else "trackpy"
    return backend


def bootstrap_label(
    video_path, split, tier=None,
    model_path=None, backend="auto", tracker="bytetrack", clip_id=None,
    diameter=DEFAULT_DIAMETER, minmass=DEFAULT_MINMASS,
    search_range=DEFAULT_SEARCH_RANGE, memory=DEFAULT_MEMORY, maxsize=DEFAULT_MAXSIZE,
):
    video_path = Path(video_path)
    resolved = _resolve_backend(backend, model_path)

    if resolved == "yolo" and not model_path:
        raise ValueError("backend='yolo' needs a model_path")

    initial_csv = video_path.parent / f"{video_path.stem}_bootstrap.csv"

    if resolved == "yolo":
        print(f"Tracking {video_path} with {model_path} ({tracker})...")
        raw = run_tracking(model_path, video_path, tracker=tracker)
        canonical = from_bytetrack_output(raw)
    else:
        print(f"Tracking {video_path} with trackpy (backend=trackpy; "
              f"diameter={diameter}, minmass={minmass}, search_range={search_range}, memory={memory})...")
        canonical = run_trackpy_tracking(
            video_path, diameter=diameter, minmass=minmass,
            search_range=search_range, memory=memory, maxsize=maxsize,
        )

    save_tracks(canonical, initial_csv)
    n_sperm = canonical["sperm"].nunique() if len(canonical) else 0
    print(f"Pre-populated {len(canonical)} rows across {n_sperm} predicted tracks -> {initial_csv}")

    print("Opening labeler for correction. Save before closing to keep your edits.")
    corrected_csv = launch_labeler_and_wait(video_path, initial_csv)

    if corrected_csv is None:
        print(f"No corrected file found -- nothing was kept, so nothing is being registered.\n"
              f"To pick up where you left off, re-run this command, or open the labeler directly:\n"
              f"  python -m data_engine.stages.label.labeler_gui "
              f"--videofile {video_path} --csvfile {initial_csv}")
        return None

    clip = ingest(video_path, corrected_csv, tier, "real", split,
                  clip_id=clip_id, label_status="model_bootstrap+corrected")
    print(f"Registered corrected clip: {clip.clip_id}")
    return clip


def _cli():
    parser = argparse.ArgumentParser(
        description="Bootstrap-label a video: initial tracker -> human correction -> registered clip"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val"])
    parser.add_argument("--tier", default=None,
                         help="low|medium|high|very_high -- optional, auto-computed from the "
                              "corrected labels' measured density if omitted (recommended)")
    parser.add_argument("--model", default=None, help="Required for --backend yolo (or auto, if given)")
    parser.add_argument("--backend", choices=["auto", "yolo", "trackpy"], default="auto",
                         help="auto (default): yolo if --model given, else trackpy")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort"], default="bytetrack",
                         help="Only used for --backend yolo")
    parser.add_argument("--clip-id", default=None)
    trackpy_group = parser.add_argument_group("trackpy tuning (only used for --backend trackpy)")
    trackpy_group.add_argument("--diameter", type=int, default=DEFAULT_DIAMETER)
    trackpy_group.add_argument("--minmass", type=float, default=DEFAULT_MINMASS)
    trackpy_group.add_argument("--search-range", type=int, default=DEFAULT_SEARCH_RANGE)
    trackpy_group.add_argument("--memory", type=int, default=DEFAULT_MEMORY)
    trackpy_group.add_argument("--maxsize", type=float, default=DEFAULT_MAXSIZE)
    args = parser.parse_args()

    bootstrap_label(
        args.video, args.split, tier=args.tier,
        model_path=args.model, backend=args.backend, tracker=args.tracker, clip_id=args.clip_id,
        diameter=args.diameter, minmass=args.minmass, search_range=args.search_range,
        memory=args.memory, maxsize=args.maxsize,
    )


if __name__ == "__main__":
    _cli()
