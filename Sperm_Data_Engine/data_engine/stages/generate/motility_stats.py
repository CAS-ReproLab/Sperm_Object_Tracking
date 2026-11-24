"""
motility_stats.py -- headless motility-statistics computation, ported and
adapted from SyntheticSimulatorCode's Param_fit (VAP, BCF, angular velocity,
sigma_theta, sigma_r, and the sum-of-two-Gaussians OMEGA fit) and Stats'
calculate_ALH_stats (amplitude of lateral head displacement -- a better,
more standard source for wobble_delta than a derived VCL-VAP heuristic;
adapted here onto canonical column names and Param_fit's own x_avg/y_avg
rather than duplicating Stats' redundant path-averaging logic).

Operates on the canonical frame,sperm,x,y dataframe format directly --
Param_fit's own configurable attribute names (x_coord='x', y_coord='y',
Agent_ID='sperm') already matched this, so no renaming was needed there;
only Stats' ALH algorithm (hardcoded to x_coord/y_coord/Agent_ID/Time) and
Param_fit's own 'Time' column needed adapting.

Pure functions only: no matplotlib, no file I/O, no `self.df` mutation --
each function takes a dataframe and returns one with new column(s) added,
so fit_params.py (real data) and validate_params.py (synthetic data) can
run the identical code path on both.
"""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from shapely.geometry import LineString

STAT_COLUMNS = ["VAP", "ALH_mean", "BCF", "OMEGA", "fitted_sigma_theta", "sigma_r", "angular_fit_SSR"]


def _ensure_time(df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    if "Time" not in df.columns:
        df = df.copy()
        df["Time"] = df["frame"] / frame_rate
    return df


def compute_vap(df: pd.DataFrame, pixel_size: float, frame_rate: float,
                 window_size: int = None, window_duration_s: float = 0.75,
                 min_periods: int = 1) -> pd.DataFrame:
    """Per-agent average path velocity (VAP), plus x_avg/y_avg (the smoothed
    path ALH and BCF both need). Ported from Param_fit.calculate_VAP.

    DATA-ENGINE FIX: window_size now derives from window_duration_s (a fixed
    ~0.75s span, converted to frames via frame_rate) rather than the
    hardcoded window_size=3 frames Param_fit used. That 3-frame default is
    only a meaningful low-pass filter at the (presumably higher) frame rate
    Param_fit was written against; at this dataset's 9fps it's only ~0.33s
    -- barely different from the raw path -- and confirmed empirically to
    let near-Nyquist artifact wiggle through as spurious BCF "beat
    crossings" (fitted BCF averaged ~5.5Hz, above the 4.5Hz Nyquist limit
    for 9fps video). Sweeping window size showed BCF/ALH converge close to
    the simulator's original hand-tuned defaults (BCF~3Hz, ALH~1.0um) around
    a ~0.75-0.8s window, and true flagellar beat frequency (typically well
    above 4.5Hz) can't be recovered at 9fps regardless of window choice --
    this picks a physically-motivated, frame-rate-independent smoothing
    span rather than a frame count tuned to one specific video's fps.
    Pass window_size explicitly to override."""
    if window_size is None:
        window_size = max(3, int(round(window_duration_s * frame_rate)))
        if window_size % 2 == 0:
            window_size += 1  # keep centered rolling symmetric
    df = _ensure_time(df, frame_rate).copy()
    vap_values = {}
    x_avg_all = pd.Series(index=df.index, dtype=float)
    y_avg_all = pd.Series(index=df.index, dtype=float)

    for agent_id, group in df.groupby("sperm"):
        group_sorted = group.sort_values("Time")
        x_avg = group_sorted["x"].rolling(window=window_size, min_periods=min_periods, center=True).mean()
        y_avg = group_sorted["y"].rolling(window=window_size, min_periods=min_periods, center=True).mean()
        x_avg_all.loc[group_sorted.index] = x_avg.values
        y_avg_all.loc[group_sorted.index] = y_avg.values

        distances = np.sqrt(np.diff(x_avg) ** 2 + np.diff(y_avg) ** 2) * pixel_size
        total_distance = np.nansum(distances)
        total_time = group_sorted["Time"].iloc[-1] - group_sorted["Time"].iloc[0]
        vap_values[agent_id] = (total_distance / total_time) if total_time > 0 else 0.0

    df["x_avg"] = x_avg_all
    df["y_avg"] = y_avg_all
    df["VAP"] = df["sperm"].map(vap_values)
    return df


def compute_alh(df: pd.DataFrame, pixel_size: float) -> pd.DataFrame:
    """Per-agent mean amplitude of lateral head displacement (ALH_mean):
    perpendicular distance from each raw-path segment to the corresponding
    smoothed-path midpoint. Requires x_avg/y_avg (compute_vap must run
    first). Adapted from Stats.calculate_ALH_stats onto canonical columns;
    that version operates in raw pixel coordinates, so pixel_size scaling
    is added here for consistency with Param_fit's VAP/BCF (both already
    scale by pixel_size)."""
    df = df.copy()
    alh_mean_values = {}

    for agent_id, group in df.groupby("sperm"):
        group_sorted = group.sort_values("Time").reset_index(drop=True)
        x_coords, y_coords = group_sorted["x"], group_sorted["y"]
        x_avg, y_avg = group_sorted["x_avg"], group_sorted["y_avg"]

        if len(x_avg) < 2:
            alh_mean_values[agent_id] = 0.0
            continue

        x_midpoints = (x_avg[:-1].values + x_avg[1:].values) / 2
        y_midpoints = (y_avg[:-1].values + y_avg[1:].values) / 2

        perpendicular_distances = []
        for i in range(len(x_midpoints)):
            x_mid, y_mid = x_midpoints[i], y_midpoints[i]
            x1, y1 = x_coords.iloc[i], y_coords.iloc[i]
            x2, y2 = x_coords.iloc[i + 1], y_coords.iloc[i + 1]
            segment_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if segment_length == 0:
                continue
            distance = abs((x2 - x1) * (y1 - y_mid) - (y2 - y1) * (x1 - x_mid)) / segment_length
            perpendicular_distances.append(distance)

        alh_mean_values[agent_id] = (np.mean(perpendicular_distances) * pixel_size) if perpendicular_distances else 0.0

    df["ALH_mean"] = df["sperm"].map(alh_mean_values)
    return df


def _segments_intersect(x1, y1, x2, y2, xa1, ya1, xa2, ya2) -> bool:
    line1 = LineString([(x1, y1), (x2, y2)])
    line2 = LineString([(xa1, ya1), (xa2, ya2)])
    return line1.crosses(line2) or line1.touches(line2)


def compute_bcf(df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Per-agent beat-cross frequency (BCF): how often the raw path crosses
    its own smoothed path, per second. Requires x_avg/y_avg (compute_vap
    must run first). Ported from Param_fit.calculate_BCF."""
    df = _ensure_time(df, frame_rate).copy()
    bcf_values = {}

    for agent_id, group in df.groupby("sperm"):
        group_sorted = group.sort_values("Time")
        x_coords, y_coords = group_sorted["x"].values, group_sorted["y"].values
        x_avg, y_avg = group_sorted["x_avg"].values, group_sorted["y_avg"].values

        if len(x_coords) < 2:
            bcf_values[agent_id] = 0.0
            continue

        num_crossings = 0
        for i in range(len(x_coords) - 1):
            if _segments_intersect(x_coords[i], y_coords[i], x_coords[i + 1], y_coords[i + 1],
                                    x_avg[i], y_avg[i], x_avg[i + 1], y_avg[i + 1]):
                num_crossings += 1

        time_values = group_sorted["Time"].values
        total_time = np.sum(np.diff(time_values)) if len(time_values) > 1 else 0
        bcf_values[agent_id] = (num_crossings / total_time) if total_time > 0 else 0.0

    df["BCF"] = df["sperm"].map(bcf_values)
    return df


def compute_angular_velocity(df: pd.DataFrame, frame_rate: float) -> pd.DataFrame:
    """Per-frame instantaneous angular velocity (rad/s), computed off the
    smoothed path (x_avg/y_avg from compute_vap, which must run first) --
    NOT the raw per-frame x/y Param_fit.calculate_angular_velocity used.

    DATA-ENGINE FIX: confirmed empirically that raw frame-to-frame angles
    are dominated by centroid-detection noise, not genuine heading change.
    Real frame-to-frame steps are tiny at this frame rate (median 2.5px,
    25% under 0.23px) -- comparable to ordinary pixel-level jitter -- and
    |angular_velocity| computed from raw x/y was strongly anti-correlated
    with step size (r=-0.42): steps under 2px showed a median |angular
    velocity| of 9.0 rad/s and a 95th percentile of 26.2, vs. 0.39/1.5 for
    20-50px steps. That noise was propagating straight into the
    fit_angular_velocity_dist() OMEGA/sigma_theta population fit, producing
    physically implausible spreads regardless of downstream trimming/
    filtering (both tried, neither fixed it -- this was the actual cause).
    ALH and BCF already avoid this exact problem by using the smoothed path;
    this made angular velocity consistent with that established pattern
    instead of the odd one out still using raw coordinates."""
    df = _ensure_time(df, frame_rate).copy()
    angular_vel_values = {}

    for agent_id, group in df.groupby("sperm"):
        group_sorted = group.sort_values("Time")
        x, y = group_sorted["x_avg"].values, group_sorted["y_avg"].values
        dx, dy = np.diff(x), np.diff(y)
        time_diffs = np.diff(group_sorted["Time"].values)
        thetas = np.arctan2(dy, dx)

        angular_velocity = [np.nan]
        for i in range(1, len(thetas)):
            if time_diffs[i - 1] > 0:
                angle_diff = thetas[i] - thetas[i - 1]
                angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # normalize to (-pi, pi)
                av_i = angle_diff / time_diffs[i - 1]
            else:
                av_i = np.nan
            angular_velocity.append(av_i)

        angular_vel_values.update({idx: val for idx, val in zip(group_sorted.index, angular_velocity)})

    df["angular_velocity"] = df.index.map(angular_vel_values)
    return df


def compute_sigma_theta(df: pd.DataFrame) -> pd.DataFrame:
    """Per-agent raw stdev of angular velocity. Requires compute_angular_velocity
    first. Ported from Param_fit.calculate_sigma_theta."""
    df = df.copy()
    stdev_values = df.groupby("sperm")["angular_velocity"].std()
    df["sigma_theta"] = df["sperm"].map(stdev_values)
    return df


def _sum_of_two_gaussians(d_theta, A1, A2, OMEGA, omega, fitted_sigma_theta):
    gauss1 = A1 * np.exp(-(d_theta - (OMEGA - omega)) ** 2 / (2 * fitted_sigma_theta ** 2))
    gauss2 = A2 * np.exp(-(d_theta - (OMEGA + omega)) ** 2 / (2 * fitted_sigma_theta ** 2))
    return gauss1 + gauss2


def fit_angular_velocity_dist(df: pd.DataFrame, min_traj_len: int = 25) -> pd.DataFrame:
    """Per-agent OMEGA + fitted_sigma_theta via a sum-of-two-Gaussians fit
    to that agent's angular-velocity histogram, symmetrically offset around
    OMEGA. Requires compute_angular_velocity first. Trajectories shorter
    than min_traj_len, or where the fit doesn't converge, get NaN. Ported
    from Param_fit.fit_angular_velocity_dist."""
    df = df.copy()
    fitted_params = {}
    ssr_values = {}

    for agent_id, group in df.groupby("sperm"):
        group_sorted = group.sort_values("Time")
        angular_velocities = group_sorted["angular_velocity"].dropna().values

        if len(angular_velocities) < min_traj_len:
            fitted_params[agent_id] = [np.nan] * 5
            ssr_values[agent_id] = np.nan
            continue

        hist, bin_edges = np.histogram(angular_velocities, bins=20, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        initial_guess = [0.5, 0.5, np.mean(angular_velocities), 0.0, np.std(angular_velocities)]

        try:
            opt_params, _ = curve_fit(
                _sum_of_two_gaussians, bin_centers, hist, p0=initial_guess,
                bounds=([0, 0, -np.inf, -np.inf, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]),
            )
            fitted_params[agent_id] = opt_params
            # Sum of squared residuals -- goodness-of-fit. `except RuntimeError`
            # below only catches outright non-convergence; it does NOT catch a
            # fit that converges to a poor/nonsensical solution on a noisy or
            # short real track (observed in practice: population OMEGA std
            # dominated by a handful of such fits, though their input trajectory
            # was too short/noisy to be rejected by min_traj_len alone). SSR
            # lets fit_params.py filter those out before pooling.
            fitted_y = _sum_of_two_gaussians(bin_centers, *opt_params)
            ssr_values[agent_id] = float(np.sum((hist - fitted_y) ** 2))
        except RuntimeError:
            fitted_params[agent_id] = [np.nan] * 5
            ssr_values[agent_id] = np.nan

    param_names = ["A1", "A2", "OMEGA", "omega", "fitted_sigma_theta"]
    for i, name in enumerate(param_names):
        df[name] = df["sperm"].map(lambda x: fitted_params.get(x, [np.nan] * 5)[i])
    df["angular_fit_SSR"] = df["sperm"].map(ssr_values)
    return df


def compute_instantaneous_radial_velocity(df: pd.DataFrame, pixel_size: float, frame_rate: float) -> pd.DataFrame:
    """Per-frame point-to-point velocity (um/s), computed off the smoothed
    path (x_avg/y_avg from compute_vap, which must run first) -- NOT the
    raw per-frame x/y Param_fit.calculate_instantaneous_radial_velocity used.

    DATA-ENGINE FIX: same noise-dominance problem confirmed for
    compute_angular_velocity applies here too -- real frame-to-frame steps
    are tiny at this frame rate (median 2.5px), so raw point-to-point speed
    is also noise-dominated, not just heading. Switched for consistency with
    that fix and with ALH/BCF, which already use the smoothed path."""
    df = _ensure_time(df, frame_rate).copy()
    velocity_values = {}

    for agent_id, group in df.groupby("sperm"):
        group_sorted = group.sort_values("Time")
        distances = np.sqrt(np.diff(group_sorted["x_avg"]) ** 2 + np.diff(group_sorted["y_avg"]) ** 2) * pixel_size
        time_diffs = np.diff(group_sorted["Time"].values)

        velocity = [np.nan]
        for i in range(1, len(distances)):
            v_i = distances[i] / time_diffs[i] if time_diffs[i - 1] > 0 else np.nan
            velocity.append(v_i)

        velocity_values.update({idx: val for idx, val in zip(group_sorted.index, velocity)})

    df["velocity"] = df.index.map(velocity_values)
    return df


def compute_sigma_r(df: pd.DataFrame) -> pd.DataFrame:
    """Per-agent stdev of instantaneous radial velocity -- the amplitude for
    Simulation's translational-noise term. Requires
    compute_instantaneous_radial_velocity first. Ported from
    Param_fit.calculate_sigma_r."""
    df = df.copy()
    stdev_values = df.groupby("sperm")["velocity"].std()
    df["sigma_r"] = df["sperm"].map(stdev_values)
    return df


def compute_all_stats(df: pd.DataFrame, pixel_size: float, frame_rate: float,
                       min_traj_len: int = 25) -> pd.DataFrame:
    """Run the full pipeline in dependency order and return one row per
    sperm with VAP, ALH_mean, BCF, OMEGA, fitted_sigma_theta, sigma_r --
    everything fit_params.py/validate_params.py need, on real or synthetic
    data alike."""
    df = _ensure_time(df, frame_rate)
    df = compute_vap(df, pixel_size, frame_rate)
    df = compute_alh(df, pixel_size)
    df = compute_bcf(df, frame_rate)
    df = compute_angular_velocity(df, frame_rate)
    df = fit_angular_velocity_dist(df, min_traj_len=min_traj_len)
    df = compute_instantaneous_radial_velocity(df, pixel_size, frame_rate)
    df = compute_sigma_r(df)

    return df.drop_duplicates(subset="sperm")[["sperm"] + STAT_COLUMNS].reset_index(drop=True)
