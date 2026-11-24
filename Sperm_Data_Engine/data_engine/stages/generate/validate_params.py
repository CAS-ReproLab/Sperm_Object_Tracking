"""
validate_params.py -- does fitting param_specs from real data actually
produce more realistic synthetic motion than the simulator's original
hand-tuned defaults?

Generates two synthetic samples (the simulator's original DEFAULT_PARAM_SPECS,
and a fit_params.py fitted param_specs.json), scores both with the exact
same motility_stats.py functions used on real data (no risk of a second,
subtly inconsistent implementation skewing the comparison), and reports
per-stat Earth Mover's Distance (scipy.stats.wasserstein_distance -- the
same technique UnifiedFramework/experiment1.ipynb already uses to compare
tracking algorithms, here repurposed to compare motion realism) against the
real pooled sample fit_params.py already computed and stored.

Run as a module from the repo root:

    python -m data_engine.stages.generate.validate_params --fit-dir data_engine/data/fitted_params/<timestamp>
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wasserstein_distance

from data_engine.common.trajectory_io import load_tracks
from data_engine.stages.generate.fit_params import PARAM_MAP
from data_engine.stages.generate.motility_stats import compute_all_stats
from data_engine.stages.generate.synthesize import DEFAULT_PARAM_SPECS, synthesize


def _synthetic_stats(num_agents, frames, pixel_size, frame_rate, param_specs, out_dir, seed):
    _video_path, csv_path = synthesize(
        str(num_agents), frames=frames, frame_rate=frame_rate, pixel_size=pixel_size,
        out_dir=out_dir, param_specs=param_specs, seed=seed,
    )
    df = load_tracks(csv_path)
    return compute_all_stats(df, pixel_size, frame_rate)


def validate_params(fit_dir, num_agents=100, frames=270, seed=0, out_dir=None):
    fit_dir = Path(fit_dir)
    with open(fit_dir / "fit_manifest.json") as f:
        manifest = json.load(f)
    with open(fit_dir / "param_specs.json") as f:
        fitted_specs = json.load(f)

    pixel_size = manifest["pixel_size"]
    frame_rate = manifest["frame_rate"]
    real_values = manifest["raw_pooled_values"]

    if out_dir is None:
        out_dir = fit_dir / "validation"
    out_dir = Path(out_dir)

    print(f"Generating synthetic sample with the simulator's DEFAULT param_specs "
          f"(num_agents={num_agents}, frames={frames})...")
    default_stats = _synthetic_stats(num_agents, frames, pixel_size, frame_rate,
                                      DEFAULT_PARAM_SPECS, out_dir / "default", seed)

    print("Generating synthetic sample with the FITTED param_specs...")
    fitted_stats = _synthetic_stats(num_agents, frames, pixel_size, frame_rate,
                                     fitted_specs, out_dir / "fitted", seed)

    rows = []
    for spec_key, (stat_col, _dist, convert) in PARAM_MAP.items():
        real_sample = real_values.get(spec_key, [])
        if not real_sample:
            continue

        default_sample = default_stats[stat_col].dropna().values
        fitted_sample = fitted_stats[stat_col].dropna().values
        # real_sample is already unit-converted (fit_params.py applies `convert`
        # before storing raw_pooled_values) -- apply the same conversion to the
        # synthetic samples, which come straight from motility_stats' raw column.
        if convert is not None:
            default_sample = convert(default_sample)
            fitted_sample = convert(fitted_sample)

        emd_default = wasserstein_distance(real_sample, default_sample) if len(default_sample) else float("nan")
        emd_fitted = wasserstein_distance(real_sample, fitted_sample) if len(fitted_sample) else float("nan")
        rows.append({
            "param": spec_key, "stat": stat_col, "n_real": len(real_sample),
            "n_default_synth": len(default_sample), "n_fitted_synth": len(fitted_sample),
            "EMD_vs_default": emd_default, "EMD_vs_fitted": emd_fitted,
            "fitted_improved": bool(emd_fitted < emd_default) if not pd.isna(emd_default) and not pd.isna(emd_fitted) else None,
        })

    table = pd.DataFrame(rows)
    print("\n=== EMD: real vs. default-synthetic vs. fitted-synthetic (lower = more realistic) ===")
    print(table.to_string(index=False))

    table.to_csv(out_dir / "emd_comparison.csv", index=False)
    print(f"\nWrote {out_dir / 'emd_comparison.csv'}")
    return table


def _cli():
    parser = argparse.ArgumentParser(
        description="Compare real vs. default-synthetic vs. fitted-synthetic motility stats via EMD"
    )
    parser.add_argument("--fit-dir", required=True, help="Directory containing fit_params.py's output")
    parser.add_argument("--num-agents", type=int, default=100)
    parser.add_argument("--frames", type=int, default=270)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    validate_params(args.fit_dir, num_agents=args.num_agents, frames=args.frames,
                     seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    _cli()
