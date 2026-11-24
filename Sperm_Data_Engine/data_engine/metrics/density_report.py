"""
density_report.py -- aggregate per-clip metrics_v2.py results by density tier.

Expects a directory of CSVs, one per validation clip, each named "<clip_id>.csv"
and shaped like metrics_v2.py's output (columns: Metric, Unfiltered, Filtered;
rows: DET, TRA, LNK, TF, MOTA, IDF1, HOTA, ...). <clip_id> must match a clip_id
in the registry so its density tier can be looked up.

Run as a module from the repo root, e.g.:

    python -m data_engine.metrics.density_report --results-dir path/to/per_clip_csvs
    python -m data_engine.metrics.density_report --results-dir path/to/csvs --filtered --out tier_summary.csv
"""
import argparse
from pathlib import Path

import pandas as pd

from data_engine.registry.registry import Registry

OVERALL_LABEL = "All Test Videos"  # summary_df's aggregate row, matching the paper's table row name


def load_clip_results(results_dir, use_filtered: bool = False) -> dict:
    """Return {clip_id: {metric: value}} for every <clip_id>.csv in results_dir."""
    col = "Filtered" if use_filtered else "Unfiltered"
    out = {}
    for csv_path in sorted(Path(results_dir).glob("*.csv")):
        clip_id = csv_path.stem
        df = pd.read_csv(csv_path)
        out[clip_id] = dict(zip(df["Metric"], df[col]))
    return out


def build_density_report(clip_results: dict, registry: Registry, include_overall: bool = True):
    """
    clip_results: {clip_id: {metric: value}}
    Returns (per_clip_df, summary_df). summary_df has one row per density
    tier, columns = mean of each metric across that tier's clips, plus
    n_clips. Clips not found in the registry are grouped under tier
    "unknown" so they're visible rather than silently dropped.

    include_overall=True (default) appends one more row, OVERALL_LABEL,
    averaging every clip directly (not the per-tier means, so a tier with
    more clips is weighted accordingly) -- the "all test videos together"
    row the paper's results tables report alongside the per-tier breakdown.
    It's appended after sorting so it always lands last regardless of how
    tier names happen to sort alphabetically.
    """
    rows = []
    for clip_id, metrics in clip_results.items():
        clip = registry.get(clip_id)
        tier = clip.density_tier if clip is not None else "unknown"
        rows.append({"clip_id": clip_id, "density_tier": tier, **metrics})

    per_clip = pd.DataFrame(rows)
    metric_cols = [c for c in per_clip.columns if c not in ("clip_id", "density_tier")]

    summary = (
        per_clip.groupby("density_tier")[metric_cols]
        .mean()
        .assign(n_clips=per_clip.groupby("density_tier").size())
        .reset_index()
        .sort_values("density_tier")
    )
    if include_overall and not per_clip.empty:
        overall = {"density_tier": OVERALL_LABEL, "n_clips": len(per_clip),
                   **per_clip[metric_cols].mean().to_dict()}
        summary = pd.concat([summary, pd.DataFrame([overall])], ignore_index=True)
    return per_clip, summary


def interleave_filtered(unfiltered_df, filtered_df, key_cols, suffix=" (filt)"):
    """Merge two same-shaped tables into one wide table with each metric's
    unfiltered value immediately followed by its filtered value, e.g.
    MOTA | MOTA (filt) | IDF1 | IDF1 (filt). key_cols (e.g. clip_id /
    density_tier, plus n_clips) are kept once at the left."""
    keys = [c for c in key_cols if c in unfiltered_df.columns]
    metrics = [c for c in unfiltered_df.columns if c not in keys]
    merged = unfiltered_df.merge(filtered_df[keys + metrics], on=keys, suffixes=("", suffix), how="left")
    ordered = list(keys)
    for m in metrics:
        ordered += [m, m + suffix]
    return merged[[c for c in ordered if c in merged.columns]]


def _cli():
    parser = argparse.ArgumentParser(description="Aggregate metrics_v2.py results by density tier")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--filtered", action="store_true",
                         help="Use the 'Filtered' column instead of 'Unfiltered'")
    parser.add_argument("--out", default=None, help="Optional path to save the per-tier summary CSV")
    args = parser.parse_args()

    registry = Registry.load()
    clip_results = load_clip_results(args.results_dir, use_filtered=args.filtered)
    if not clip_results:
        raise SystemExit(f"No *.csv files found in {args.results_dir}")

    per_clip, summary = build_density_report(clip_results, registry)

    print("=== Per-clip ===")
    print(per_clip.to_string(index=False))
    print("\n=== Per density tier (mean) ===")
    print(summary.to_string(index=False))

    if args.out:
        summary.to_csv(args.out, index=False)
        print(f"\nSaved tier summary to {args.out}")


if __name__ == "__main__":
    _cli()
