"""
fit_params.py -- fit Simulation's param_specs distributions from every
registered real clip's ground truth, pooled together (per the user: not
scoped per-tier).

Per-parameter mapping (see data_engine/README.md for the full reasoning):
  OMEGA        <- fit_angular_velocity_dist()'s per-agent OMEGA (direct)
  sigma_theta  <- that same fit's fitted_sigma_theta (more refined than the raw stdev)
  v            <- VAP (average path velocity -- smoothed, better forward-speed proxy than raw VCL)
  wobble_w     <- BCF (beat-cross frequency) * 2*pi
  wobble_delta <- ALH_mean (amplitude of lateral head displacement -- the standard CASA term for this)
  sigma_r      <- raw stdev of instantaneous radial velocity (now meaningful --
                   see the DATA-ENGINE FIX in stages/generate/simulator/simulation.py)
  wobble_phi   -- left as Uniform(0, 2*pi); an arbitrary recording-start phase,
                   not something real data should inform

Distribution families match the original hand-tuned example in
synthesize.py's DEFAULT_PARAM_SPECS (normal for OMEGA/wobble_w, lognormal
for the rest) rather than doing automatic family selection -- there's
little real data to fit against, so trusting the original author's modeling
choices and only re-fitting the actual parameters is the more defensible
choice.

Run as a module from the repo root:

    python -m data_engine.stages.generate.fit_params
"""
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_engine.common.trajectory_io import load_tracks
from data_engine.registry.registry import REPO_ROOT, Registry
from data_engine.stages.generate import calibrate
from data_engine.stages.generate.motility_stats import compute_all_stats
from data_engine.stages.generate.synthesize import DEFAULT_PARAM_SPECS
from data_engine.stages.validate.tracking_metrics import filter_stationary

# Param_fit's own defaults ("currently for 5X mag"). CONFIRM these actually
# match how the real clips were recorded before trusting the fit -- every
# physical-unit value here is systematically mis-scaled otherwise, and this
# can't be verified from the code alone.
DEFAULT_PIXEL_SIZE = 1.8  # um/pixel
DEFAULT_FRAME_RATE = 9    # fps

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "data" / "fitted_params"

# param_specs key -> (motility_stats column, distribution family, unit conversion)
PARAM_MAP = {
    "OMEGA": ("OMEGA", "normal", None),
    "sigma_theta": ("fitted_sigma_theta", "lognormal", None),
    "v": ("VAP", "lognormal", None),
    "wobble_w": ("BCF", "normal", lambda x: 2 * np.pi * x),
    "wobble_delta": ("ALH_mean", "lognormal", None),
    "sigma_r": ("sigma_r", "lognormal", None),
}

# OMEGA and sigma_theta both come from fit_angular_velocity_dist()'s per-agent
# curve fit, which can converge to a poor/nonsensical solution on a noisy or
# short real track without raising (see motility_stats.py's angular_fit_SSR).
# Percentile-trimming alone wasn't enough to keep this from dominating the
# population estimate -- confirmed empirically via validate_params.py, worse
# EMD than the untouched defaults on every parameter before this filter was
# added. Dropping the worst-fitting quarter of agents (by SSR) before pooling
# targets the actual cause instead.
SSR_FILTERED_PARAMS = {"OMEGA", "sigma_theta"}
SSR_KEEP_QUANTILE = 0.75

# v/sigma_r: what's measured off a trajectory (VAP, stdev of instantaneous
# radial velocity) is a biased, though monotonic, function of the true
# simulator input -- confirmed empirically (see calibrate.py's module
# docstring). These get their real-data targets corrected through an
# inverted calibration curve before fitting.
CALIBRATED_PARAMS = {"v", "sigma_r"}

# wobble_w/wobble_delta: BCF/ALH's true-input -> measured-output mapping is
# NON-monotonic (aliasing fold-over past this dataset's ~4.5Hz Nyquist
# limit at 9fps, confirmed empirically) -- not invertible, so no
# calibration curve can correct them. Skip fitting these from real data
# entirely and keep the simulator's original hand-tuned defaults.
DEFAULTED_PARAMS = {"wobble_w", "wobble_delta"}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _clean_values(values, dist):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if dist == "lognormal":
        values = values[values > 0]
    return values


def _fit_distribution(values, dist):
    """
    values: already-cleaned (no NaN, positive-only if lognormal) array.

    Trims to the [1st, 99th] percentile range before fitting mean/sd, not
    just for the clip bounds. Found empirically necessary: a handful of
    individual per-agent curve_fit()s in fit_angular_velocity_dist()
    converge to physically nonsensical values on noisy/short real tracks
    (the `except RuntimeError` guard there only catches outright
    non-convergence, not "converged to nonsense"), and unweighted mean/std
    let a few such outliers dominate the population estimate -- e.g. OMEGA's
    median was 0.03 rad/s (matching the original hand-tuned assumption of
    0.06 almost exactly) while its untrimmed std was 28, produced entirely
    by a small tail out to +/-150-240. Confirmed via validate_params.py:
    fitting without this trim made EMD *worse* than the untouched defaults
    on every single parameter.
    """
    if len(values) == 0:
        return None
    lo, hi = float(np.percentile(values, 1)), float(np.percentile(values, 99))
    trimmed = values[(values >= lo) & (values <= hi)]
    if len(trimmed) == 0:
        trimmed = values
    if dist == "normal":
        return {"dist": "normal", "mean": float(np.mean(trimmed)), "sd": float(np.std(trimmed)), "clip": [lo, hi]}
    if dist == "lognormal":
        log_values = np.log(trimmed)
        return {"dist": "lognormal", "mean_log": float(np.mean(log_values)), "sd_log": float(np.std(log_values)),
                "clip": [lo, hi]}
    raise ValueError(f"Unsupported distribution family: {dist}")


def fit_params(pixel_size=DEFAULT_PIXEL_SIZE, frame_rate=DEFAULT_FRAME_RATE, out_dir=None):
    registry = Registry.load()
    real_clips = [c for c in registry.clips if c.source_type == "real"]
    if not real_clips:
        raise ValueError("No real clips registered -- nothing to fit from")

    per_clip_stats = []
    used_clips, skipped_clips = [], []
    for clip in real_clips:
        if not clip.ground_truth_csv:
            skipped_clips.append(clip.clip_id)
            continue
        df = load_tracks(REPO_ROOT / clip.ground_truth_csv)
        df = filter_stationary(df)
        if df.empty:
            skipped_clips.append(clip.clip_id)
            continue
        stats = compute_all_stats(df, pixel_size, frame_rate)
        stats["clip_id"] = clip.clip_id
        per_clip_stats.append(stats)
        used_clips.append(clip.clip_id)

    if not per_clip_stats:
        raise ValueError("No usable real clips (all skipped -- missing ground truth, or empty after filtering)")

    pooled = pd.concat(per_clip_stats, ignore_index=True)

    ssr = pooled["angular_fit_SSR"].dropna()
    ssr_cutoff = float(ssr.quantile(SSR_KEEP_QUANTILE)) if len(ssr) else None
    well_fit = pooled[pooled["angular_fit_SSR"] <= ssr_cutoff] if ssr_cutoff is not None else pooled
    if ssr_cutoff is not None:
        print(f"SSR filter for OMEGA/sigma_theta: keeping {len(well_fit)}/{len(pooled)} agents "
              f"(SSR <= {ssr_cutoff:.4g}, the {SSR_KEEP_QUANTILE:.0%} quantile)")

    calibration_curves = {}
    for spec_key in CALIBRATED_PARAMS:
        print(f"Building calibration curve for {spec_key!r} "
              f"(probing {calibrate.DEFAULT_PROBE_VALUES[spec_key]})...")
        calibration_curves[spec_key] = calibrate.build_calibration_curve(spec_key)
        print(calibration_curves[spec_key].to_string(index=False))

    param_specs = {}
    sample_info = {}
    raw_pooled_values = {}
    for spec_key, (stat_col, dist, convert) in PARAM_MAP.items():
        if spec_key in DEFAULTED_PARAMS:
            # Non-invertible calibration (aliasing) -- keep the raw measured
            # real values around for validate_params.py's diagnostic EMD
            # table, but don't fit from them; use the simulator's own
            # hand-tuned default spec unchanged.
            values = pooled[stat_col].values
            if convert is not None:
                values = convert(values)
            cleaned = _clean_values(values, dist)
            raw_pooled_values[spec_key] = [float(v) for v in cleaned]
            param_specs[spec_key] = DEFAULT_PARAM_SPECS[spec_key]
            sample_info[spec_key] = {"source_column": stat_col, "n_samples": len(cleaned),
                                      "fitted": DEFAULT_PARAM_SPECS[spec_key],
                                      "note": "not fit from real data -- non-invertible calibration (aliasing); "
                                              "kept the simulator's original hand-tuned default"}
            continue

        source_df = well_fit if spec_key in SSR_FILTERED_PARAMS else pooled
        values = source_df[stat_col].values
        if convert is not None:
            values = convert(values)
        cleaned = _clean_values(values, dist)
        raw_pooled_values[spec_key] = [float(v) for v in cleaned]

        if spec_key in CALIBRATED_PARAMS and len(cleaned):
            fit_input = calibrate.invert(calibration_curves[spec_key], cleaned)
        else:
            fit_input = cleaned

        spec = _fit_distribution(fit_input, dist)
        if spec is None:
            print(f"WARNING: no usable samples for {spec_key!r} (from {stat_col!r}) -- omitted from param_specs")
            continue
        param_specs[spec_key] = spec
        sample_info[spec_key] = {"source_column": stat_col, "n_samples": len(cleaned), "fitted": spec}
        if spec_key in CALIBRATED_PARAMS:
            sample_info[spec_key]["note"] = "fit from calibration-corrected values, not raw measured values -- see calibrate.py"

    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = DEFAULT_OUT_ROOT / stamp
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    param_specs_path = out_dir / "param_specs.json"
    with open(param_specs_path, "w") as f:
        json.dump(param_specs, f, indent=2)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "pixel_size": pixel_size,
        "frame_rate": frame_rate,
        "clip_ids_used": used_clips,
        "clip_ids_skipped": skipped_clips,
        "n_agents_pooled": int(len(pooled)),
        "sample_info": sample_info,
        "raw_pooled_values": raw_pooled_values,
    }
    with open(out_dir / "fit_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Fit from {len(used_clips)} real clip(s): {used_clips}")
    if skipped_clips:
        print(f"Skipped {len(skipped_clips)} clip(s) (missing ground truth or empty after filtering): {skipped_clips}")
    print(f"{len(pooled)} agent(s) pooled")
    for key, info in sample_info.items():
        print(f"  {key}: n={info['n_samples']}, fitted={info['fitted']}")

    print(f"\nWrote {param_specs_path}")
    print(f"Wrote {out_dir / 'fit_manifest.json'}")
    return out_dir


def _cli():
    parser = argparse.ArgumentParser(description="Fit Simulation's param_specs from every registered real clip")
    parser.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE,
                         help="um/pixel -- confirm this matches how the real clips were actually recorded")
    parser.add_argument("--frame-rate", type=float, default=DEFAULT_FRAME_RATE, help="fps")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    fit_params(pixel_size=args.pixel_size, frame_rate=args.frame_rate, out_dir=args.out_dir)


if __name__ == "__main__":
    _cli()
