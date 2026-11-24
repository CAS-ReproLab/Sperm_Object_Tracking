"""
trackpy_tracker.py -- the trackpy-based fallback tracker: what the original
videos were actually started with (UnifiedFramework/tracker.py +
UnifiedFramework/main.py's default_config), vendored here as a small
wrapper rather than a separate stage, per the user's framing: this is a
default tracker bootstrap_label.py can always fall back to when no trained
YOLO model exists yet, not a distinct pipeline stage.

Only determineCentroids() + trackCentroids() are reused -- segmentCells()
(segmentation masks) is skipped: irrelevant to centroid-only bootstrap
tracks, and doubles processing time.

Unlike the YOLO/ByteTrack path, trackCentroids() already renames trackpy's
`particle` column to `sperm` and has `frame, sperm, x, y` present alongside
trackpy's own extra columns (mass, size, ecc, ...) -- no format conversion
needed before this is ready for the labeler or trajectory_io.
"""
import trackpy as tp

from data_engine.stages.label.utils import loadVideo

# UnifiedFramework/main.py's default_config -- confirmed to be what actually
# processed the original videos, not tracker.py's own function defaults
# (diameter=7, minmass=100) or processVideo()'s hardcoded values (5, 50, 5),
# both of which differ from this.
DEFAULT_DIAMETER = 11
DEFAULT_MINMASS = 500
DEFAULT_SEARCH_RANGE = 21
DEFAULT_MEMORY = 3
# main.py never overrides this, so the original videos were processed with
# tracker.py's determineCentroids()'s own default -- matched explicitly here
# rather than relying on trackpy's own tp.batch() default, which differs.
DEFAULT_MAXSIZE = 5


def run_trackpy_tracking(
    video_path,
    diameter: int = DEFAULT_DIAMETER,
    minmass: float = DEFAULT_MINMASS,
    search_range: int = DEFAULT_SEARCH_RANGE,
    memory: int = DEFAULT_MEMORY,
    maxsize: float = DEFAULT_MAXSIZE,
):
    frames = loadVideo(str(video_path), as_gray=True)

    tp.quiet()  # suppress trackpy's per-frame progress spam
    f = tp.batch(frames, diameter=diameter, minmass=minmass, maxsize=maxsize)
    t = tp.link(f, search_range=search_range, memory=memory, adaptive_stop=0.2, adaptive_step=0.95)
    t = tp.filter_stubs(t, 15)
    t = t.rename(columns={"particle": "sperm"})
    t = t.reset_index(drop=True)
    return t
