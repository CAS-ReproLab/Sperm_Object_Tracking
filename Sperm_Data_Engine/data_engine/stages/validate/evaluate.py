"""
evaluate.py -- run tracking + metrics scoring over every val clip in the
registry (or a --tiers-filtered subset), then roll the results up by density
tier via data_engine/metrics/density_report.py.

One command produces both the per-clip <clip_id>.csv files (metrics_v2.py's
exact output shape) and the density-level summary table -- Phase 1's
density_report.py, reused directly rather than re-implemented.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.validate.evaluate --model training_weights/finetune_1/weights/best.pt
    python -m data_engine.stages.validate.evaluate --model best.pt --tiers 40 --tracker botsort
    python -m data_engine.stages.validate.evaluate --tracker trackpy   # baseline, no --model

--tracker trackpy scores the Trackpy baseline instead of a model (no --model
needed) -- same per-clip/density-tier output shape, so its results drop
straight into density_report.interleave_filtered/comparison tables alongside
a model's for a trackpy-vs-model report.
"""
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_engine.common.trajectory_io import load_tracks
from data_engine.metrics.density_report import build_density_report, interleave_filtered
from data_engine.registry.registry import Registry
from data_engine.stages.validate.track import run_tracking
from data_engine.stages.validate.tracking_metrics import score_tracks

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACKER_OUTPUTS_DIR = REPO_ROOT / "data_engine" / "data" / "tracker_outputs"


def evaluate(model_path=None, tiers=None, tracker="bytetrack", out_dir=None):
    """tracker="trackpy" scores the Trackpy baseline instead of a model --
    model_path is unused (and must be omitted) in that case."""
    if tracker == "trackpy":
        if model_path:
            raise ValueError("model_path must not be given when tracker='trackpy'")
    elif not model_path:
        raise ValueError("model_path is required unless tracker='trackpy'")

    registry = Registry.load()
    clips = registry.clips_matching(split="val")
    if tiers:
        clips = [c for c in clips if c.density_tier in tiers]
    if not clips:
        raise ValueError(f"No val clips matched tiers={tiers}")

    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = (TRACKER_OUTPUTS_DIR / f"trackpy_eval_{stamp}" if tracker == "trackpy"
                   else Path(model_path).resolve().parent / f"eval_{stamp}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped, failed = [], []
    for clip in clips:
        if not clip.source_video or not clip.ground_truth_csv:
            skipped.append(clip.clip_id)
            continue

        video_path = REPO_ROOT / clip.source_video
        gt_path = REPO_ROOT / clip.ground_truth_csv

        try:
            print(f"[{clip.clip_id}] tracking ({tracker})...")
            if tracker == "trackpy":
                from data_engine.stages.label.trackpy_tracker import run_trackpy_tracking
                pred = run_trackpy_tracking(video_path)
            else:
                pred = run_tracking(model_path, video_path, tracker=tracker)
            gt = load_tracks(gt_path)

            print(f"[{clip.clip_id}] scoring...")
            results = score_tracks(pred, gt)
            results.to_csv(out_dir / f"{clip.clip_id}.csv", index=False)
        except Exception as e:
            # A bad ground-truth CSV (e.g. a duplicate sperm ID from an
            # unsplit crossover) shouldn't throw away every other clip's
            # already-computed results -- record it and keep going, same as
            # the missing-file skip above.
            print(f"[{clip.clip_id}] FAILED: {e}")
            failed.append((clip.clip_id, str(e)))

    if skipped:
        print(f"Skipped {len(skipped)} clip(s) missing source_video/ground_truth_csv: {skipped}")
    if failed:
        print(f"\nFailed {len(failed)} clip(s) -- excluded from the summary below:")
        for clip_id, err in failed:
            print(f"  {clip_id}: {err}")
        (out_dir / "_failed_clips.txt").write_text(
            "\n".join(f"{clip_id}: {err}" for clip_id, err in failed))

    print(f"\nPer-clip results in: {out_dir}")

    clip_results, clip_results_f = {}, {}
    for csv_path in out_dir.glob("*.csv"):
        if csv_path.name.startswith("_"):
            continue
        df = pd.read_csv(csv_path)
        clip_results[csv_path.stem] = dict(zip(df["Metric"], df["Unfiltered"]))
        clip_results_f[csv_path.stem] = dict(zip(df["Metric"], df["Filtered"]))

    if not clip_results:
        raise ValueError(
            f"Every matched val clip failed or was skipped (skipped={skipped}, "
            f"failed={[c for c, _ in failed]}) -- nothing to summarize.")

    per_clip, summary = build_density_report(clip_results, registry)
    per_clip_f, summary_f = build_density_report(clip_results_f, registry)
    print("\n=== Per-clip (unfiltered) ===")
    print(per_clip.to_string(index=False))
    print("\n=== Per density tier (mean; unfiltered | filtered = stationary sperm removed) ===")
    both = interleave_filtered(summary, summary_f, ["density_tier", "n_clips"])
    print(both.to_string(index=False))
    # Unfiltered file keeps its original name/shape (Run History reads it);
    # the filtered and side-by-side versions are new siblings.
    summary.to_csv(out_dir / "_density_tier_summary.csv", index=False)
    summary_f.to_csv(out_dir / "_density_tier_summary_filtered.csv", index=False)
    both.to_csv(out_dir / "_density_tier_summary_both.csv", index=False)
    interleave_filtered(per_clip, per_clip_f, ["clip_id", "density_tier"]).to_csv(
        out_dir / "_per_clip_both.csv", index=False)

    return out_dir


def _cli():
    parser = argparse.ArgumentParser(description="Evaluate a model (or the Trackpy baseline) against every registered val clip")
    parser.add_argument("--model", default=None, help="Required unless --tracker trackpy")
    parser.add_argument("--tiers", action="append", default=None,
                         help="Filter to this density tier (repeatable; default: all val clips)")
    parser.add_argument("--tracker", choices=["bytetrack", "botsort", "trackpy"], default="bytetrack")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()
    if args.tracker != "trackpy" and not args.model:
        parser.error("--model is required unless --tracker trackpy")

    evaluate(args.model, tiers=args.tiers, tracker=args.tracker, out_dir=args.out_dir)


if __name__ == "__main__":
    _cli()
