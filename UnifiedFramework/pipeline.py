from PyQt6.QtCore import QThread, QObject, pyqtSignal
import traceback

import utils
from plugins.preprocessors import PREPROCESSORS, apply_fill_holes
from plugins.perception import PERCEPTION_METHODS, PRODUCES_SEGMENTATION
from plugins.trackers import TRACKERS, REQUIRES_SEGMENTATION
from plugins.postprocessors import POSTPROCESSORS


STAGE_ORDER = ["preprocessing", "perception", "tracking", "postprocessing"]


class PipelineWorker(QThread):
    stage_started   = pyqtSignal(str)           # stage name
    stage_completed = pyqtSignal(str, object)   # stage name, result DataFrame/array
    error_occurred  = pyqtSignal(str)           # traceback string

    def __init__(self, frames, config, stop_after=None, parent=None):
        super().__init__(parent)
        self.frames     = frames
        self.config     = config
        self.stop_after = stop_after  # "preprocessing" | "perception" | None (run all)

    def run(self):
        try:
            config = self.config
            frames = self.frames

            self.stage_started.emit("preprocessing")
            preprocessed = PREPROCESSORS[config["preprocessor"]](frames, config)
            if config.get("fill_holes", False):
                preprocessed = apply_fill_holes(preprocessed, config)
            self.stage_completed.emit("preprocessing", preprocessed)
            if self.stop_after == "preprocessing":
                return

            self.stage_started.emit("perception")
            detections = PERCEPTION_METHODS[config["perception"]](preprocessed, config)
            self.stage_completed.emit("perception", detections)
            if self.stop_after == "perception":
                return

            self.stage_started.emit("tracking")
            tracks = TRACKERS[config["tracker"]](detections, config)
            self.stage_completed.emit("tracking", tracks)

            for pp_name in config.get("postprocessors", []):
                if pp_name == "none":
                    continue
                self.stage_started.emit(f"postprocessing:{pp_name}")
                tracks = POSTPROCESSORS[pp_name](tracks, preprocessed, config)
                self.stage_completed.emit(f"postprocessing:{pp_name}", tracks)

        except Exception:
            self.error_occurred.emit(traceback.format_exc())


class Pipeline(QObject):
    """
    Owns pipeline state (frames, detections, tracks) and a worker thread.
    GUI components connect to these signals to react to pipeline progress.
    """
    stage_started   = pyqtSignal(str)
    stage_completed = pyqtSignal(str, object)
    error_occurred  = pyqtSignal(str)
    finished        = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frames              = None
        self.preprocessed_frames = None
        self.detections          = None
        self.tracks              = None
        self._worker             = None

    # ── public API ────────────────────────────────────────────────────────────

    def load_video(self, path):
        """Load video frames into memory. Returns the frame array."""
        self.frames = utils.loadVideo(path, as_gray=True)
        # Reset previous results
        self.preprocessed_frames = None
        self.detections          = None
        self.tracks              = None
        return self.frames

    def run(self, config, stop_after=None):
        """Validate config then kick off the worker thread.

        stop_after: None (full pipeline) | "preprocessing" | "perception"
        """
        if self.frames is None:
            raise RuntimeError("No video loaded.")

        perception = config.get("perception", "trackpy")
        tracker    = config.get("tracker",    "trackpy")
        if REQUIRES_SEGMENTATION.get(tracker) and not PRODUCES_SEGMENTATION.get(perception):
            raise RuntimeError(
                f"Tracker '{tracker}' requires segmentation masks, "
                f"but perception method '{perception}' does not produce them.\n"
                f"Choose a perception method that produces segmentations, or a different tracker."
            )

        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()

        self._worker = PipelineWorker(self.frames, config, stop_after=stop_after)
        self._worker.stage_started.connect(self._on_stage_started)
        self._worker.stage_completed.connect(self._on_stage_completed)
        self._worker.error_occurred.connect(self.error_occurred)
        self._worker.finished.connect(self.finished)
        self._worker.start()

    def stop(self):
        if self._worker and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()

    # ── internal ──────────────────────────────────────────────────────────────

    def _on_stage_started(self, stage):
        self.stage_started.emit(stage)

    def _on_stage_completed(self, stage, result):
        if stage == "preprocessing":
            self.preprocessed_frames = result
        elif stage == "perception":
            self.detections = result
        elif stage == "tracking" or stage.startswith("postprocessing:"):
            self.tracks = result
        self.stage_completed.emit(stage, result)
