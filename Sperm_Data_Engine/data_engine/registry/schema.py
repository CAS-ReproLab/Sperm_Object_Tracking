"""
schema.py -- one record per video clip in the data-engine registry.

A "clip" is one video's worth of already-processed YOLO training data: a
directory containing images/ and labels/ (the same layout data_process.py and
data_process_validate.py in the root reference code have always produced).
The registry is what ties a set of these directories together into a
dataset, instead of hand-editing sperm_dataset.yaml's train/val lists.
"""
from dataclasses import asdict, dataclass
from typing import Optional

VALID_SOURCE_TYPES = {"real", "synthetic", "augmented"}
VALID_SPLITS = {"train", "val"}
VALID_LABEL_STATUS = {"human", "model_bootstrap", "model_bootstrap+corrected"}

# Real clips are binned by measured sperm-per-frame density (see
# registry.py's DENSITY_BINS / bin_real_density()) -- a fixed vocabulary
# independent of frame rate/resolution, unlike a raw count. Synthetic clips
# use a numeric string tier instead: that number IS the simulator's
# num_agents (the generation target), not a bin, so it isn't drawn from a
# fixed set -- any positive-integer string is valid.
REAL_DENSITY_TIERS = {"low", "medium", "high", "very_high", "unknown"}


def _is_valid_density_tier(value: str) -> bool:
    return value in REAL_DENSITY_TIERS or (value.isdigit() and int(value) > 0)


# Kept for any code that still wants the old fixed-set check on legacy tiers.
VALID_TIERS = REAL_DENSITY_TIERS | {"10", "40", "160", "640"}


@dataclass
class Clip:
    clip_id: str
    path: str                              # relative to repo root; must contain images/ and labels/
    source_type: str                       # real | synthetic | augmented
    density_tier: str                      # real: "low"|"medium"|"high"|"very_high"|"unknown"
                                            # synthetic: a numeric string -- the generation num_agents
    split: str                             # train | val
    label_status: str = "human"
    bootstrap_model_id: Optional[str] = None
    added_in_iteration: int = 0
    notes: str = ""
    source_video: Optional[str] = None     # relative path to the clip's source .mp4, if known
    ground_truth_csv: Optional[str] = None # relative path to its frame,sperm,x,y[,label] CSV, if known
    measured_density: Optional[float] = None  # avg unique sperm/frame from ground_truth_csv, if measured
                                               # (real clips: what density_tier's bin was computed from --
                                               # kept alongside the bin so re-binning after a boundary
                                               # change never needs re-measuring every clip)

    def __post_init__(self):
        if self.source_type not in VALID_SOURCE_TYPES:
            raise ValueError(f"{self.clip_id}: invalid source_type {self.source_type!r}")
        if not _is_valid_density_tier(self.density_tier):
            raise ValueError(f"{self.clip_id}: invalid density_tier {self.density_tier!r}")
        if self.split not in VALID_SPLITS:
            raise ValueError(f"{self.clip_id}: invalid split {self.split!r}")
        if self.label_status not in VALID_LABEL_STATUS:
            raise ValueError(f"{self.clip_id}: invalid label_status {self.label_status!r}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Clip":
        return cls(**d)
