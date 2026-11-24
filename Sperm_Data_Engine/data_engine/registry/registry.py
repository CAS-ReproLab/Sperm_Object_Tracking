"""
registry.py -- the data-engine's single source of truth for which video clips
exist, what density tier / source they are, and which split they belong to.

Run as a module from the repo root, e.g.:

    python -m data_engine.registry.registry seed
    python -m data_engine.registry.registry list --tier 40 --split val
"""
import argparse
import csv
import re
from pathlib import Path

import yaml

from data_engine.registry.schema import Clip, VALID_SPLITS, VALID_TIERS

REQUIRED_TRACK_COLUMNS = {"frame", "sperm", "x", "y"}

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.yaml"
LEGACY_DATA_DIR = REPO_ROOT / "data"
LEGACY_YAML = REPO_ROOT / "sperm_dataset.yaml"

# Real-clip density bins, by average unique sperm per frame (not total unique
# track-ID count -- that's inflated by tracking fragmentation, especially on
# not-yet-corrected clips, and isn't the quantity synthetic tiers describe
# either; synthetic num_agents is a per-frame count). Upper bounds are
# inclusive; the last bin has no upper bound. Set 2026-09 against the then-
# current real corpus (all measured 17-77/frame) -- revisit if/when denser
# real footage actually gets recorded.
DENSITY_BINS = [
    ("low", 0, 80),
    ("medium", 81, 240),
    ("high", 241, 800),
    ("very_high", 801, None),
]


def bin_real_density(avg_per_frame: float) -> str:
    for label, lo, hi in DENSITY_BINS:
        if avg_per_frame >= lo and (hi is None or avg_per_frame <= hi):
            return label
    return "unknown"  # unreachable given the bins above, but never raise on a measurement


def measure_avg_density(ground_truth_csv: Path) -> float:
    """Average unique sperm/frame from a canonical frame,sperm,x,y CSV --
    the quantity DENSITY_BINS bins on. Imported lazily (not at module level)
    to avoid a hard pandas dependency for callers that never measure."""
    from data_engine.common.trajectory_io import load_tracks
    df = load_tracks(ground_truth_csv)
    return float(df.groupby("frame")["sperm"].nunique().mean())


def parse_density_tier(folder_name: str) -> str:
    """
    Density tier is keyed on the existing sperm-per-field naming convention
    already used in folder names (10/40/160/640 -- see _legacy_source_type()
    below for what these numbers actually are: previously-generated synthetic
    density targets, not real dilution factors). Anything else (RL_*, R_*,
    test_*, ...) is tagged "unknown" rather than guessed.
    """
    m = re.match(r"^(\d+)", folder_name)
    if m and m.group(1) in (VALID_TIERS - {"unknown"}):
        return m.group(1)
    return "unknown"


class Registry:
    def __init__(self, clips=None):
        self.clips: list[Clip] = clips or []

    # ---- persistence ----

    @classmethod
    def load(cls, path=MANIFEST_PATH) -> "Registry":
        path = Path(path)
        if not path.exists():
            return cls([])
        with open(path) as f:
            data = yaml.safe_load(f) or []
        return cls([Clip.from_dict(d) for d in data])

    def save(self, path=MANIFEST_PATH) -> None:
        with open(path, "w") as f:
            f.write("# Data-engine clip registry. Generated/edited via\n")
            f.write("# data_engine/registry/registry.py -- avoid hand-editing where possible.\n")
            yaml.safe_dump([c.to_dict() for c in self.clips], f, sort_keys=False)

    # ---- queries ----

    def clips_matching(self, tier=None, split=None, source_type=None) -> list[Clip]:
        result = self.clips
        if tier is not None:
            result = [c for c in result if c.density_tier == tier]
        if split is not None:
            result = [c for c in result if c.split == split]
        if source_type is not None:
            result = [c for c in result if c.source_type == source_type]
        return result

    def get(self, clip_id: str) -> Clip | None:
        for c in self.clips:
            if c.clip_id == clip_id:
                return c
        return None

    def add_clip(self, clip: Clip, replace: bool = False) -> None:
        existing = self.get(clip.clip_id)
        if existing is not None:
            if not replace:
                raise ValueError(f"clip_id {clip.clip_id!r} already registered")
            self.clips.remove(existing)
        self.clips.append(clip)

    def set_split(self, clip_id: str, split: str) -> None:
        """Move a clip between train and val (registry edit only)."""
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
        clip = self.get(clip_id)
        if clip is None:
            raise KeyError(f"clip_id {clip_id!r} not registered")
        clip.split = split

    def unregister(self, clip_id: str) -> None:
        """Take a clip off the registry WITHOUT touching its files, so it can
        be re-ingested later (unlike the GUI's delete, which removes files)."""
        clip = self.get(clip_id)
        if clip is None:
            raise KeyError(f"clip_id {clip_id!r} not registered")
        self.clips.remove(clip)


def _read_legacy_split_map(yaml_path=LEGACY_YAML) -> dict:
    """{folder_name: 'train'|'val'} from the root sperm_dataset.yaml's train:/val:
    lists (each entry looks like '10_1/images')."""
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    split_map = {}
    for split in ("train", "val"):
        for entry in cfg.get(split, []) or []:
            folder = entry.split("/")[0]
            split_map[folder] = split
    return split_map


def _csv_has_track_columns(csv_path: Path) -> bool:
    try:
        with open(csv_path, newline="") as f:
            header = next(csv.reader(f))
    except (StopIteration, OSError):
        return False
    return REQUIRED_TRACK_COLUMNS.issubset({c.strip() for c in header})


def detect_source_video(clip_dir: Path):
    """Return the one *.mp4 directly inside clip_dir, relative to REPO_ROOT, or None."""
    videos = sorted(clip_dir.glob("*.mp4"))
    if len(videos) == 1:
        return str(videos[0].relative_to(REPO_ROOT)).replace("\\", "/")
    return None


def detect_ground_truth_csv(clip_dir: Path, clip_id: str):
    """
    Prefer a *trackpy_format*.csv (the converted canonical format). Fall back
    to <clip_id>.csv if it already has frame,sperm,x,y columns -- true for
    the oldest clips (10, 40), which predate that naming convention.
    """
    trackpy_matches = sorted(clip_dir.glob("*trackpy_format*.csv"))
    if len(trackpy_matches) == 1:
        return str(trackpy_matches[0].relative_to(REPO_ROOT)).replace("\\", "/")

    fallback = clip_dir / f"{clip_id}.csv"
    if fallback.exists() and _csv_has_track_columns(fallback):
        return str(fallback.relative_to(REPO_ROOT)).replace("\\", "/")

    return None


def _legacy_source_type(folder: str, density_tier: str) -> str:
    """
    Per direct confirmation from the professor (2026-09-02): despite the
    dilution-style names, the 10/40/160/640-family clips in this legacy
    dataset are previously-generated SYNTHETIC videos, not real recordings --
    the RL_* clips are the actual real recordings. This is empirical
    knowledge about this specific 22-clip legacy set (fixed by the current
    sperm_dataset.yaml), not a general naming rule -- it is NOT used to infer
    source_type for anything added later. ingest.py always takes an explicit
    --source for new clips, real or otherwise, regardless of naming.
    """
    return "synthetic" if density_tier != "unknown" else "real"


def scan_legacy_data(data_dir=LEGACY_DATA_DIR, yaml_path=LEGACY_YAML, iteration: int = 0) -> Registry:
    """
    One-time seed: for every folder referenced in the root sperm_dataset.yaml's
    train:/val: lists that actually has images/+labels/ on disk, register it
    (source_type per _legacy_source_type() above; label_status="human" since
    all of it -- real or previously-generated synthetic -- has trustworthy,
    non-model-predicted ground truth). Folders that exist on disk with
    images/+labels/ but are NOT referenced by the yaml are reported, not
    registered -- they're leftover experiment/inference-output folders, not
    part of the active dataset (the root data/ folder has many of these).
    """
    split_map = _read_legacy_split_map(yaml_path)
    registry = Registry.load()

    on_disk_yolo_folders = {
        p.name for p in Path(data_dir).iterdir()
        if p.is_dir() and (p / "images").is_dir() and (p / "labels").is_dir()
    }

    skipped_not_in_yaml = sorted(on_disk_yolo_folders - set(split_map))

    registered, missing_on_disk, preserved = [], [], []
    no_video, no_gt_csv = [], []
    for folder, split in sorted(split_map.items()):
        if folder not in on_disk_yolo_folders:
            missing_on_disk.append(folder)
            continue

        clip_dir = Path(data_dir) / folder
        source_video = detect_source_video(clip_dir)
        ground_truth_csv = detect_ground_truth_csv(clip_dir, folder)
        if source_video is None:
            no_video.append(folder)
        if ground_truth_csv is None:
            no_gt_csv.append(folder)

        # Real clips carrying measured_density have already been through the
        # 2026-09 density-tier reorg (re-registered by hand with a proper
        # low/medium/high/very_high bin -- see data_engine/README.md's "Data
        # reorganization" section). sperm_dataset.yaml still names these
        # folders directly (RL_1, etc.), so a naive re-seed would silently
        # clobber that back to density_tier="unknown". Preserve it instead.
        existing = registry.get(folder)
        if existing is not None and existing.measured_density is not None:
            preserved.append(folder)
            continue

        density_tier = parse_density_tier(folder)
        clip = Clip(
            clip_id=folder,
            path=str(clip_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
            source_type=_legacy_source_type(folder, density_tier),
            density_tier=density_tier,
            split=split,
            label_status="human",
            added_in_iteration=iteration,
            notes="seeded from legacy data/ + sperm_dataset.yaml",
            source_video=source_video,
            ground_truth_csv=ground_truth_csv,
        )
        registry.add_clip(clip, replace=True)
        registered.append(folder)

    registry.save()

    print(f"Registered {len(registered)} clips.")
    if preserved:
        print(f"Preserved {len(preserved)} clip(s) already re-registered by the density-tier reorg "
              f"(not overwritten): {preserved}")
    if missing_on_disk:
        print(f"WARNING: {len(missing_on_disk)} folder(s) listed in sperm_dataset.yaml "
              f"but missing images/+labels/ on disk: {missing_on_disk}")
    if no_video:
        print(f"WARNING: {len(no_video)} clip(s) have no unambiguous source .mp4 "
              f"(0 or >1 found): {no_video}")
    if no_gt_csv:
        print(f"WARNING: {len(no_gt_csv)} clip(s) have no detected ground-truth CSV: {no_gt_csv}")
    if skipped_not_in_yaml:
        print(f"Note: {len(skipped_not_in_yaml)} folder(s) on disk have images/+labels/ but "
              f"aren't in sperm_dataset.yaml's train/val lists, so were not registered: "
              f"{skipped_not_in_yaml}")
    return registry


def _cli():
    parser = argparse.ArgumentParser(description="Data-engine clip registry")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("seed", help="Scan legacy data/ + sperm_dataset.yaml and populate manifest.yaml")

    list_p = sub.add_parser("list", help="List registered clips")
    list_p.add_argument("--tier", default=None)
    list_p.add_argument("--split", default=None)
    list_p.add_argument("--source", default=None)

    split_p = sub.add_parser("set-split", help="Move a clip between train and val")
    split_p.add_argument("clip_id")
    split_p.add_argument("split", choices=["train", "val"])
    unreg_p = sub.add_parser("unregister", help="Take a clip off the registry (files are kept)")
    unreg_p.add_argument("clip_id")

    args = parser.parse_args()

    if args.command == "seed":
        scan_legacy_data()
    elif args.command in ("set-split", "unregister"):
        registry = Registry.load()
        if args.command == "set-split":
            registry.set_split(args.clip_id, args.split)
        else:
            registry.unregister(args.clip_id)
        registry.save()
        print("OK:", args.command, args.clip_id)
    elif args.command == "list":
        registry = Registry.load()
        clips = registry.clips_matching(tier=args.tier, split=args.split, source_type=args.source)
        for c in clips:
            print(f"{c.clip_id:20s} tier={c.density_tier:8s} split={c.split:6s} "
                  f"source={c.source_type:10s} status={c.label_status}")
        print(f"\n{len(clips)} clip(s)")


if __name__ == "__main__":
    _cli()
