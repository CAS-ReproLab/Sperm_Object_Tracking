"""
synthetic_showcase.py -- a deliberately-staged synthetic clip + 2-panel
figure for the paper, showing the range of motion behaviors the Langevin
model can produce in one video: a few "ballistic" sperm (fast, near-straight
paths), a few "hyperactive"/diffusive sperm (lots of turning, large lateral
excursions, little net progress), and the rest sampled from the real fitted
population distribution (fitted_params/<stamp>/param_specs.json) -- the
"most sperm are somewhere in between" middle ground.

This does NOT reuse synthesize.py directly: synthesize() samples ONE
param_specs for an entire population, so there's no way to force a handful
of agents into specific extreme regimes while sampling the rest normally.
Instead this runs three separate data_engine/stages/generate/simulator
Simulation()s (one per regime, same frame_rate/pixel_size/width/height/
num_frames so their `positions` arrays line up step-for-step), concatenates
their position arrays into one combined array, and renders that through the
same Generator used everywhere else -- so the combined video looks exactly
like any other synthetic clip, just with its motion regimes hand-picked.

Run as a module from the repo root, e.g.:

    python -m data_engine.analysis.synthetic_showcase \\
        --out-dir data_engine/data/synthetic_showcase \\
        --figure data_engine/data/visualizations/synthetic_showcase_figure.png
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_engine.stages.generate.simulator.image_generator import Generator
from data_engine.stages.generate.simulator.simulation import Simulation

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = REPO_ROOT / "data_engine" / "data" / "synthetic_showcase"

# Fitted population distribution -- same file the real training data was
# generated from. "Typical" sperm are sampled from this, same as any other
# synthetic clip.
_FITTED_PARAMS_DIR = REPO_ROOT / "data_engine" / "data" / "fitted_params"


def _latest_fitted_param_specs():
    runs = sorted(_FITTED_PARAMS_DIR.glob("*/param_specs.json"))
    if not runs:
        raise FileNotFoundError(
            f"No fitted param_specs.json under {_FITTED_PARAMS_DIR} -- run fit_params.py first, "
            f"or pass --typical-param-specs explicitly.")
    return runs[-1]


# Hand-picked, deliberately extreme (not fitted) regimes for the showcase
# groups -- scalars, not distributions, so every agent in a group looks the
# same way rather than averaging out the effect we're trying to show.
# Reference points: the latest fit's clip ranges are roughly v=[5,100],
# sigma_theta=[0.13,4.8], sigma_r=[1,26] (see fitted_params/*/param_specs.json).
BALLISTIC_PARAMS = {
    "v": 55.0,              # fast, well above the fitted median (~25 um/s)
    "OMEGA": 0.0,            # no net chirality drift -- stays straight over the window
    "sigma_theta": 0.08,     # near the low end -- very little random turning
    "sigma_r": 1.0,          # clip floor -- minimal positional jitter
}
HYPERACTIVE_PARAMS = {
    "v": 40.0,               # still fast (real hyperactivated motility IS fast), but...
    "OMEGA": 0.0,
    "sigma_theta": 4.0,      # ...near the high end: heavy random turning dominates net progress
    "sigma_r": 10.0,         # extra positional jitter on top
    "wobble_delta": 4.0,     # bigger lateral excursions (clip ceiling is 6.0)
}


def _make_group(num_agents, num_frames, frame_rate, pixel_size, width, height, param_specs, seed=None):
    if seed is not None:
        np.random.seed(seed)
    sim = Simulation(
        num_agents=num_agents, num_frames=num_frames, frame_rate=frame_rate,
        pixel_size=pixel_size, width=width, height=height, param_specs=param_specs,
    )
    sim.run_simulation()
    return sim


def render_showcase(
    out_dir=None, figure_path=None,
    n_ballistic=3, n_hyperactive=3, n_typical=6,
    frames=120, frame_rate=9.0, pixel_size=1.8, width=569, height=569,
    mid_seconds=10.0, typical_param_specs=None, seed=0, original_label="Example Frame from Synthetic Video",
):
    out_dir = Path(out_dir) if out_dir else DEFAULT_OUT_ROOT
    out_dir.mkdir(parents=True, exist_ok=True)

    if typical_param_specs is None:
        typical_param_specs = _latest_fitted_param_specs()
    typical_param_specs = Path(typical_param_specs)
    with open(typical_param_specs) as f:
        typical_specs = json.load(f)

    groups = [
        ("ballistic", n_ballistic, BALLISTIC_PARAMS, seed + 1),
        ("hyperactive", n_hyperactive, HYPERACTIVE_PARAMS, seed + 2),
        ("typical", n_typical, typical_specs, seed + 3),
    ]

    sims = []
    for label, n, specs, group_seed in groups:
        if n <= 0:
            continue
        sim = _make_group(n, frames, frame_rate, pixel_size, width, height, specs, seed=group_seed)
        sims.append((label, sim))

    # sim.positions is at the raw physics resolution (dt=0.01s/step -- e.g.
    # 2410 steps for a 220-frame, 9fps request), NOT one row per video frame.
    # Downsample by each sim's own steps_per_frame first (identical across
    # groups, since they share frame_rate/dt), exactly like
    # Simulation.save_trajectories() does -- otherwise "frame" in the
    # rendered video/tracks is actually a physics step, ~steps_per_frame times
    # slower than the requested frame_rate (a real bug caught by comparing a
    # ballistic sperm's actual displacement against its configured speed).
    downsampled = []
    for _, sim in sims:
        idx = np.arange(0, sim.num_steps, sim.steps_per_frame)
        downsampled.append(sim.positions[idx])

    combined_positions = np.concatenate(downsampled, axis=1)
    agent_size = sims[0][1].get_agent_size()
    num_frames_out = combined_positions.shape[0]

    generator = Generator(combined_positions, agent_size, sims[0][1].width, sims[0][1].height,
                           num_frames_out, frame_rate=frame_rate)
    video_path = out_dir / "showcase_video.mp4"
    generator.save_video(str(video_path))

    # Build one canonical frame,sperm,x,y,group dataframe across all groups,
    # from the same downsampled (one-row-per-video-frame) arrays used above.
    rows = []
    next_id = 0
    for (label, _sim), positions in zip(sims, downsampled):
        n_agents = positions.shape[1]
        for local_id in range(n_agents):
            xs = positions[:, local_id, 0]
            ys = positions[:, local_id, 1]
            valid = ~np.isnan(xs) & ~np.isnan(ys)
            frame_idx = np.nonzero(valid)[0]
            if len(frame_idx) < 2:
                next_id += 1
                continue
            rows.append(pd.DataFrame({
                "frame": frame_idx, "sperm": next_id, "x": xs[valid], "y": ys[valid], "group": label,
            }))
            next_id += 1
    tracks = pd.concat(rows, ignore_index=True)
    tracks_path = out_dir / "showcase_tracks.csv"
    tracks.to_csv(tracks_path, index=False)

    if figure_path is None:
        figure_path = out_dir / "showcase_figure.png"
    figure_path = render_showcase_figure(
        video_path, tracks, figure_path, mid_frame=round(mid_seconds * frame_rate), mid_seconds=mid_seconds,
        original_label=original_label,
    )

    print(f"Video: {video_path}")
    print(f"Tracks: {tracks_path}")
    print(f"Figure: {figure_path}")
    return video_path, tracks_path, figure_path


_GROUP_COLORS = {"ballistic": "#1f77b4", "hyperactive": "#d62728", "typical": "#7f7f7f"}
_GROUP_LABELS = {"ballistic": "Ballistic-like", "hyperactive": "Diffusive-like", "typical": "Typical (fitted population)"}


def render_showcase_figure(video_path, tracks, out_path, mid_frame, mid_seconds=10.0,
                            original_label="Example Frame from Synthetic Video", linewidth=1.3, dpi=200):
    """2-panel figure: frame 0 alone, and every group's tracks accumulated
    up to mid_frame, colored by group (not by individual ID, unlike
    trails_figure.py) so the three regimes read clearly at a glance."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, frame0 = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ok2, frame_mid = cap.read()
    cap.release()
    if not (ok and ok2):
        raise ValueError(f"Could not read frame 0 or frame {mid_frame} from {video_path}")
    frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    frame_mid = cv2.cvtColor(frame_mid, cv2.COLOR_BGR2GRAY)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))

    axes[0].imshow(frame0, cmap="gray", vmin=0, vmax=255)
    axes[0].set_xlabel(original_label, fontsize=13)

    axes[1].imshow(frame_mid, cmap="gray", vmin=0, vmax=255)
    sub = tracks[tracks["frame"] <= mid_frame]
    for (sperm_id, group), g in sub.groupby(["sperm", "group"]):
        g = g.sort_values("frame")
        if len(g) < 2:
            continue
        axes[1].plot(g["x"], g["y"], color=_GROUP_COLORS[group], linewidth=linewidth, alpha=0.9)
    axes[1].set_xlabel(f"Example Tracks after ~{mid_seconds:g} seconds", fontsize=13)

    handles = [plt.Line2D([0], [0], color=_GROUP_COLORS[g], lw=2, label=_GROUP_LABELS[g])
               for g in ("ballistic", "hyperactive", "typical") if g in tracks["group"].unique()]
    axes[1].legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.85)

    for ax, bg in zip(axes, (frame0, frame_mid)):
        ax.set_xlim(0, bg.shape[1])
        ax.set_ylim(bg.shape[0], 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.08, wspace=0.03)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _cli():
    parser = argparse.ArgumentParser(description="Render a staged ballistic/hyperactive/typical synthetic showcase")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--figure", default=None)
    parser.add_argument("--n-ballistic", type=int, default=3)
    parser.add_argument("--n-hyperactive", type=int, default=3)
    parser.add_argument("--n-typical", type=int, default=6)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--frame-rate", type=float, default=9.0)
    parser.add_argument("--pixel-size", type=float, default=1.8)
    parser.add_argument("--width", type=float, default=569)
    parser.add_argument("--height", type=float, default=569)
    parser.add_argument("--mid-seconds", type=float, default=10.0)
    parser.add_argument("--original-label", default="Example Frame from Synthetic Video")
    parser.add_argument("--typical-param-specs", default=None,
                         help="Default: the latest fitted_params/*/param_specs.json")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    render_showcase(
        out_dir=args.out_dir, figure_path=args.figure,
        n_ballistic=args.n_ballistic, n_hyperactive=args.n_hyperactive, n_typical=args.n_typical,
        frames=args.frames, frame_rate=args.frame_rate, pixel_size=args.pixel_size,
        width=args.width, height=args.height, mid_seconds=args.mid_seconds,
        typical_param_specs=args.typical_param_specs, seed=args.seed, original_label=args.original_label,
    )


if __name__ == "__main__":
    _cli()
