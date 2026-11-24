"""
synthesize.py -- generate a synthetic sperm-motility video + canonical
trajectory CSV at a target density tier, using SyntheticSimulatorCode's
agent-based simulator (vendored in ./simulator/).

Defaults to the simulator's original hand-tuned param_specs (copied from
SyntheticSimulatorCode/main.py's example), but --param-specs can load a
fitted set instead -- see fit_params.py, which pools every registered real
clip's motion statistics into exactly this format.

Density tier maps directly onto the simulator's num_agents: it's literally
the sperm-per-field count, the same quantity the 10/40/160/640 tier names
already describe.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.generate.synthesize --tier 40 --frames 60
    python -m data_engine.stages.generate.synthesize --tier 40 --param-specs data_engine/data/fitted_params/<timestamp>/param_specs.json
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from data_engine.stages.generate.simulator.image_generator import Generator, load_positions_from_csv
from data_engine.stages.generate.simulator.simulation import Simulation

DEFAULT_OUT_ROOT = Path(__file__).resolve().parents[2] / "data"

# SyntheticSimulatorCode/main.py's example param_specs, used as-is (see module docstring).
DEFAULT_PARAM_SPECS = {
    "v": {"dist": "lognormal", "mean_log": np.log(30.0), "sd_log": 0.25, "clip": [5, 120]},
    "OMEGA": {"dist": "normal", "mean": 0.06, "sd": 0.02, "clip": [-0.02, 0.2]},
    "sigma_theta": {"dist": "lognormal", "mean_log": np.log(0.10), "sd_log": 0.20, "clip": [0.03, 0.6]},
    "wobble_w": {"dist": "normal", "mean": 2 * np.pi * 3.0, "sd": 0.01,
                 "clip": [2 * np.pi * 1, 2 * np.pi * 8]},
    "wobble_delta": {"dist": "lognormal", "mean_log": np.log(1.0), "sd_log": 0.04, "clip": [0.1, 6.0]},
}


def _to_canonical(raw_csv_path, canonical_csv_path) -> pd.DataFrame:
    """Same rename/frame-mapping as SyntheticSimulatorCode/main.py's --convert_csv."""
    df = pd.read_csv(raw_csv_path)
    times = sorted(df["Time"].unique())
    time_to_frame = {t: f for f, t in enumerate(times)}
    df["Time"] = df["Time"].map(time_to_frame)
    df = df.rename(columns={"Time": "frame", "Agent_ID": "sperm", "x_coord": "x", "y_coord": "y"})
    df.to_csv(canonical_csv_path, index=False)
    return df


def synthesize(
    tier: str,
    frames: int = 270,
    frame_rate: float = 9,
    pixel_size: float = 1.8,
    width: float = 569,
    height: float = 569,
    seed: int = None,
    out_dir=None,
    param_specs=None,
):
    if seed is not None:
        np.random.seed(seed)

    num_agents = int(tier)

    if out_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = DEFAULT_OUT_ROOT / f"synth_{tier}_{stamp}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if param_specs is None:
        param_specs = DEFAULT_PARAM_SPECS
    elif isinstance(param_specs, (str, Path)):
        with open(param_specs) as f:
            param_specs = json.load(f)

    sim = Simulation(
        num_agents=num_agents,
        num_frames=frames,
        frame_rate=frame_rate,
        pixel_size=pixel_size,
        width=width,
        height=height,
        param_specs=param_specs,
    )
    sim.run_simulation()

    raw_csv = out_dir / "trajectories_raw.csv"
    canonical_csv = out_dir / "trajectories_trackpy_format.csv"
    sim.save_trajectories(str(raw_csv))
    canonical_df = _to_canonical(raw_csv, canonical_csv)

    positions, num_steps, _n_agents_final = load_positions_from_csv(str(raw_csv))
    generator = Generator(
        positions, sim.get_agent_size(), sim.width, sim.height, num_steps,
        frame_rate=sim.frame_rate,
    )
    video_path = out_dir / "video.mp4"
    generator.save_video(str(video_path))

    n_sperm = canonical_df["sperm"].nunique()
    print(f"Synthesized {frames} frames, requested num_agents={num_agents}, "
          f"final unique sperm IDs={n_sperm} (can exceed num_agents -- an agent "
          f"crossing the frame boundary spawns a new ID rather than wrapping in place)")
    print(f"Video: {video_path}")
    print(f"Canonical CSV: {canonical_csv}")

    return video_path, canonical_csv


def _cli():
    parser = argparse.ArgumentParser(description="Generate a synthetic sperm-motility clip")
    parser.add_argument("--tier", required=True,
                         help="num_agents to simulate -- any positive integer, not limited to the "
                              "legacy 10/40/160/640 set (e.g. use this to target the low/medium/"
                              "high/very_high real-video density bins in registry.DENSITY_BINS)")
    parser.add_argument("--frames", type=int, default=270)
    parser.add_argument("--frame-rate", type=float, default=9)
    parser.add_argument("--pixel-size", type=float, default=1.8)
    parser.add_argument("--width", type=float, default=569)
    parser.add_argument("--height", type=float, default=569)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--param-specs", default=None,
                         help="Path to a fit_params.py param_specs.json; default uses the "
                              "simulator's original hand-tuned example values")
    args = parser.parse_args()

    synthesize(
        args.tier, frames=args.frames, frame_rate=args.frame_rate,
        pixel_size=args.pixel_size, width=args.width, height=args.height,
        seed=args.seed, out_dir=args.out_dir, param_specs=args.param_specs,
    )


if __name__ == "__main__":
    _cli()
