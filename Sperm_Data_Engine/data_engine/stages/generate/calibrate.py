"""
calibrate.py -- corrects fit_params.py's real-data targets for `v` and
`sigma_r` before they're used as literal Simulation() input parameters.

Why this exists: OMEGA and sigma_theta are fit directly from Simulation's
own state variables, reconstructed per-agent via curve_fit in
motility_stats.fit_angular_velocity_dist() -- "measured value" and
"simulator input" are the same quantity there by construction. `v` and
`sigma_r` are different: what's measured off a trajectory (VAP, stdev of
instantaneous radial velocity) is a DERIVED statistic of the resulting
motion, not a direct readout of the simulator's per-agent v/sigma_r state.
Confirmed empirically (see data_engine/README.md's parameter-fitting
section) that this measurement is a systematically biased -- though
monotonic -- function of the true input: e.g. a true v=30 um/s measures
back as VAP~51 um/s. fit_params.py's original approach (feed real
MEASURED VAP directly in as `v`) made synthetic sperm swim ~1.6-1.8x
faster than real ones, which is exactly the inflated-EMD failure mode
that was observed.

This module builds a small empirical calibration curve (true input ->
measured output) by running the simulator at a handful of known input
values with every other parameter held at the simulator's own hand-tuned
defaults, then inverts it (monotonic for v/sigma_r, so ordinary 1-D
interpolation) to map each real per-agent MEASURED value back to an
estimated TRUE input value before fitting v/sigma_r's population
distribution from those corrected values instead of the raw measured ones.

wobble_w/wobble_delta are deliberately NOT handled here: their
true-input -> measured-output mapping (BCF/ALH) is non-monotonic --
an aliasing fold-over past the ~4.5Hz Nyquist limit for this dataset's
9fps video, confirmed empirically with a 5-point sweep (1/3/5/7/9 Hz true
wobble frequency measuring back as 1.9/5.6/7.6/3.9/1.5 Hz BCF -- rises,
peaks, folds back down). A non-monotonic map has no single-valued
inverse, so no calibration curve can correct it; fit_params.py falls back
to the simulator's original hand-tuned defaults for those two instead.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from data_engine.common.trajectory_io import load_tracks
from data_engine.stages.generate.motility_stats import compute_all_stats
from data_engine.stages.generate.synthesize import DEFAULT_PARAM_SPECS, synthesize

# param_specs key -> motility_stats column measured off the resulting trajectory
CALIBRATION_TARGETS = {
    "v": "VAP",
    "sigma_r": "sigma_r",
}

# True-value probe points, log-ish spaced across roughly the range real
# clips actually show for each parameter (see fit_manifest.json's clip
# bounds from an initial uncalibrated fit) -- wide enough to interpolate
# the real population's spread without extrapolating far past it.
DEFAULT_PROBE_VALUES = {
    "v": [5.0, 15.0, 30.0, 60.0, 100.0, 150.0],
    "sigma_r": [1.0, 3.0, 8.0, 15.0, 25.0, 35.0],
}

_TMP_ROOT = Path(__file__).resolve().parents[2] / "data" / "_calibration_tmp"


def _baseline_specs(exclude_key):
    """DEFAULT_PARAM_SPECS with `exclude_key` removed (the caller sets it to
    a near-deterministic probe value instead). sigma_r isn't in
    DEFAULT_PARAM_SPECS at all (the simulator's own internal fallback is a
    plain 0.0) -- add a same-shaped zero default so probing e.g. `v` doesn't
    accidentally also introduce translational noise as a confound."""
    specs = dict(DEFAULT_PARAM_SPECS)
    specs.setdefault("sigma_r", {"dist": "normal", "mean": 0.0, "sd": 0.0, "clip": [0.0, 0.0]})
    specs.pop(exclude_key, None)
    return specs


def _measure_at(param_key, stat_col, true_value, num_agents=40, frames=150,
                 frame_rate=9, pixel_size=1.8, seeds=(0, 1)):
    specs = _baseline_specs(param_key)
    specs[param_key] = {
        "dist": "normal", "mean": float(true_value), "sd": 1e-6,
        "clip": [float(true_value) - 1e-3, float(true_value) + 1e-3],
    }
    medians = []
    for seed in seeds:
        out_dir = _TMP_ROOT / f"{param_key}_{true_value}_{seed}"
        try:
            _video, csv_path = synthesize(
                str(num_agents), frames=frames, frame_rate=frame_rate, pixel_size=pixel_size,
                out_dir=out_dir, param_specs=specs, seed=seed,
            )
            df = load_tracks(csv_path)
            stats = compute_all_stats(df, pixel_size, frame_rate)
            medians.append(stats[stat_col].median())
        finally:
            shutil.rmtree(out_dir, ignore_errors=True)
    return float(np.mean(medians))


def build_calibration_curve(param_key, true_values=None, **kwargs) -> pd.DataFrame:
    """Runs the simulator at each true_value, measures the corresponding
    motility_stats column, returns a (true, measured) DataFrame sorted by
    measured value (required for np.interp-based inversion)."""
    stat_col = CALIBRATION_TARGETS[param_key]
    true_values = true_values if true_values is not None else DEFAULT_PROBE_VALUES[param_key]
    rows = [{"true": tv, "measured": _measure_at(param_key, stat_col, tv, **kwargs)} for tv in true_values]
    return pd.DataFrame(rows).sort_values("measured").reset_index(drop=True)


def invert(curve: pd.DataFrame, measured_values):
    """Maps measured values back to estimated true input values via 1-D
    linear interpolation over the calibration curve's (measured -> true)
    pairs. Flat extrapolation outside the probed range (np.interp's
    default) rather than blowing up on an out-of-range real outlier."""
    measured_values = np.asarray(measured_values, dtype=float)
    return np.interp(measured_values, curve["measured"].values, curve["true"].values)
