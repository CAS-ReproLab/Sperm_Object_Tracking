import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QPushButton, QFileDialog, QScrollArea, QSizePolicy, QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from plugins.preprocessors  import PREPROCESSORS,   PREPROCESSOR_PARAMS
from plugins.perception      import PERCEPTION_METHODS, PERCEPTION_PARAMS, PRODUCES_SEGMENTATION
from plugins.trackers        import TRACKERS,        TRACKER_PARAMS
from plugins.postprocessors  import POSTPROCESSORS,  POSTPROCESSOR_PARAMS

# Postprocessors shown in the panel (in this order)
_POSTPROCESSOR_ORDER = [k for k in POSTPROCESSORS if k != "none"]


class ControlPanel(QWidget):
    """
    Left-hand panel: algorithm selection, dynamic parameter widgets,
    global settings, and pipeline controls.
    """

    run_requested           = pyqtSignal(dict)        # emits full config dict
    preview_preprocess_requested = pyqtSignal(dict)  # stop after preprocessing
    preview_detect_requested     = pyqtSignal(dict)  # stop after perception
    video_requested         = pyqtSignal(str)        # emits chosen video path
    config_loaded           = pyqtSignal(dict)       # emits loaded config dict

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self._param_widgets   = {}   # key → widget, for all algo params
        self._pp_checkboxes   = {}   # postprocessor name → QCheckBox
        self._pp_param_groups = {}   # postprocessor name → QGroupBox (with widgets inside)

        self._build_ui()
        self._on_preprocessor_changed(0)
        self._on_perception_changed(0)
        self._on_tracker_changed(0)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # File buttons
        file_group = QGroupBox("Files")
        file_layout = QVBoxLayout(file_group)

        self._btn_load_video  = QPushButton("Load Video…")
        self._btn_load_config = QPushButton("Load Config…")
        self._btn_save_config = QPushButton("Save Config…")

        self._btn_load_video.clicked.connect(self._pick_video)
        self._btn_load_config.clicked.connect(self._pick_config)
        self._btn_save_config.clicked.connect(self._save_config)

        file_layout.addWidget(self._btn_load_video)
        file_layout.addWidget(self._btn_load_config)
        file_layout.addWidget(self._btn_save_config)
        outer.addWidget(file_group)

        # Scrollable algorithm + parameter area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        self._algo_layout = QVBoxLayout(content)
        self._algo_layout.setContentsMargins(0, 0, 0, 0)
        self._algo_layout.setSpacing(6)

        # ── Preprocessor ─────────────────────────────────────────────────────
        pre_group = QGroupBox("Preprocessor")
        pre_form  = QFormLayout(pre_group)
        self._combo_pre = QComboBox()
        self._combo_pre.addItems(list(PREPROCESSORS.keys()))
        self._combo_pre.currentIndexChanged.connect(self._on_preprocessor_changed)
        pre_form.addRow("Method:", self._combo_pre)
        self._pre_params_box = QGroupBox()
        self._pre_params_box.setFlat(True)
        self._pre_params_layout = QFormLayout(self._pre_params_box)
        pre_form.addRow(self._pre_params_box)
        self._algo_layout.addWidget(pre_group)

        # ── Perception ───────────────────────────────────────────────────────
        perc_group = QGroupBox("Perception (Detect + Segment)")
        perc_form  = QFormLayout(perc_group)
        self._combo_perc = QComboBox()
        self._combo_perc.addItems(list(PERCEPTION_METHODS.keys()))
        self._combo_perc.currentIndexChanged.connect(self._on_perception_changed)
        perc_form.addRow("Method:", self._combo_perc)
        self._perc_params_box = QGroupBox()
        self._perc_params_box.setFlat(True)
        self._perc_params_layout = QFormLayout(self._perc_params_box)
        perc_form.addRow(self._perc_params_box)
        self._algo_layout.addWidget(perc_group)

        # ── Tracker ──────────────────────────────────────────────────────────
        trk_group = QGroupBox("Tracker")
        trk_form  = QFormLayout(trk_group)
        self._combo_trk = QComboBox()
        self._combo_trk.addItems(list(TRACKERS.keys()))
        self._combo_trk.currentIndexChanged.connect(self._on_tracker_changed)
        trk_form.addRow("Method:", self._combo_trk)
        self._trk_params_box = QGroupBox()
        self._trk_params_box.setFlat(True)
        self._trk_params_layout = QFormLayout(self._trk_params_box)
        trk_form.addRow(self._trk_params_box)
        self._algo_layout.addWidget(trk_group)

        # ── Postprocessors ───────────────────────────────────────────────────
        pp_group = QGroupBox("Postprocessors")
        self._pp_layout = QVBoxLayout(pp_group)
        for pp_name in _POSTPROCESSOR_ORDER:
            chk = QCheckBox(pp_name)
            chk.setChecked(pp_name == "interpolate")
            chk.toggled.connect(lambda checked, n=pp_name: self._on_pp_toggled(n, checked))
            self._pp_checkboxes[pp_name] = chk

            params = POSTPROCESSOR_PARAMS.get(pp_name, {})
            param_box = self._make_param_group(params, prefix=f"pp_{pp_name}_")
            param_box.setVisible(chk.isChecked() and len(params) > 0)
            self._pp_param_groups[pp_name] = param_box

            self._pp_layout.addWidget(chk)
            self._pp_layout.addWidget(param_box)

        self._algo_layout.addWidget(pp_group)

        # ── Global settings ──────────────────────────────────────────────────
        global_group = QGroupBox("Global Settings")
        global_form  = QFormLayout(global_group)

        self._spin_fps = QDoubleSpinBox()
        self._spin_fps.setRange(1.0, 240.0)
        self._spin_fps.setValue(9.0)
        self._spin_fps.setSingleStep(1.0)
        global_form.addRow("FPS:", self._spin_fps)

        self._spin_px = QDoubleSpinBox()
        self._spin_px.setRange(0.01, 100.0)
        self._spin_px.setValue(1.0476)
        self._spin_px.setDecimals(4)
        self._spin_px.setSingleStep(0.01)
        global_form.addRow("Pixel size (px/μm):", self._spin_px)

        self._algo_layout.addWidget(global_group)
        self._algo_layout.addStretch()

        scroll.setWidget(content)
        outer.addWidget(scroll)

        # Preview buttons
        preview_row = QHBoxLayout()

        self._btn_preview_pre = QPushButton("Preview Preprocess")
        self._btn_preview_pre.setEnabled(False)
        self._btn_preview_pre.setToolTip("Run preprocessing only and display result in viewer")
        self._btn_preview_pre.clicked.connect(
            lambda: self.preview_preprocess_requested.emit(self.get_config())
        )

        self._btn_preview_det = QPushButton("Preview Detection")
        self._btn_preview_det.setEnabled(False)
        self._btn_preview_det.setToolTip("Run preprocessing + detection and show detections in viewer")
        self._btn_preview_det.clicked.connect(
            lambda: self.preview_detect_requested.emit(self.get_config())
        )

        preview_row.addWidget(self._btn_preview_pre)
        preview_row.addWidget(self._btn_preview_det)
        outer.addLayout(preview_row)

        # Run button
        self._btn_run = QPushButton("Run Full Pipeline")
        self._btn_run.setFixedHeight(36)
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._emit_run)
        outer.addWidget(self._btn_run)

    # ── param widget factory ──────────────────────────────────────────────────

    def _make_param_group(self, params, prefix=""):
        """Build a flat QGroupBox with a QFormLayout from a params schema dict."""
        box    = QGroupBox()
        box.setFlat(True)
        form   = QFormLayout(box)
        form.setContentsMargins(12, 0, 0, 0)

        for key, schema in params.items():
            widget = self._make_widget(schema, prefix + key)
            form.addRow(schema.get("label", key) + ":", widget)

        return box

    def _make_widget(self, schema, store_key):
        """Create the right widget type and register it in self._param_widgets."""
        kind = schema.get("type", "float")

        if kind == "int":
            w = QSpinBox()
            w.setRange(schema.get("min", 0), schema.get("max", 9999))
            w.setSingleStep(schema.get("step", 1))
            default = schema.get("default")
            w.setValue(default if default is not None else schema.get("min", 0))

        elif kind == "float":
            w = QDoubleSpinBox()
            w.setRange(schema.get("min", 0.0), schema.get("max", 9999.0))
            w.setSingleStep(schema.get("step", 0.1))
            w.setDecimals(4)
            default = schema.get("default")
            if default is None:
                w.setSpecialValueText("none")
                w.setValue(w.minimum())
            else:
                w.setValue(float(default))

        elif kind == "choice":
            w = QComboBox()
            choices = schema.get("choices", [])
            w.addItems(choices)
            default = schema.get("default", choices[0] if choices else "")
            if default in choices:
                w.setCurrentText(default)

        else:
            w = QSpinBox()

        self._param_widgets[store_key] = w
        return w

    # ── algorithm change handlers ─────────────────────────────────────────────

    def _rebuild_params(self, layout_widget, form_layout, params, prefix):
        # Remove old widgets
        keys_to_remove = [k for k in self._param_widgets if k.startswith(prefix)]
        for k in keys_to_remove:
            del self._param_widgets[k]
        while form_layout.rowCount():
            form_layout.removeRow(0)

        for key, schema in params.items():
            widget = self._make_widget(schema, prefix + key)
            form_layout.addRow(schema.get("label", key) + ":", widget)

        layout_widget.setVisible(bool(params))

    def _on_preprocessor_changed(self, _):
        name   = self._combo_pre.currentText()
        params = PREPROCESSOR_PARAMS.get(name, {})
        self._rebuild_params(self._pre_params_box, self._pre_params_layout, params, "pre_")

    def _on_perception_changed(self, _):
        name   = self._combo_perc.currentText()
        params = PERCEPTION_PARAMS.get(name, {})
        self._rebuild_params(self._perc_params_box, self._perc_params_layout, params, "perc_")

        # If this perception method already produces segmentations, disable the
        # "segment" postprocessor checkbox (it would be redundant).
        produces_seg = PRODUCES_SEGMENTATION.get(name, False)
        if "segment" in self._pp_checkboxes:
            self._pp_checkboxes["segment"].setEnabled(not produces_seg)
            if produces_seg:
                self._pp_checkboxes["segment"].setChecked(False)

    def _on_tracker_changed(self, _):
        name   = self._combo_trk.currentText()
        params = TRACKER_PARAMS.get(name, {})
        self._rebuild_params(self._trk_params_box, self._trk_params_layout, params, "trk_")

    def _on_pp_toggled(self, name, checked):
        param_box = self._pp_param_groups.get(name)
        if param_box:
            params = POSTPROCESSOR_PARAMS.get(name, {})
            param_box.setVisible(checked and len(params) > 0)

    # ── config read / write ───────────────────────────────────────────────────

    def get_config(self):
        """Collect all widget values into a config dict."""
        config = {}

        config["preprocessor"] = self._combo_pre.currentText()
        config["perception"]   = self._combo_perc.currentText()
        config["tracker"]      = self._combo_trk.currentText()
        config["fps"]          = self._spin_fps.value()
        config["pixel_size"]   = self._spin_px.value()

        config["postprocessors"] = [
            name for name in _POSTPROCESSOR_ORDER
            if self._pp_checkboxes[name].isChecked()
        ]

        for store_key, widget in self._param_widgets.items():
            # Strip prefix to get the raw config key
            for prefix in ("pre_", "perc_", "trk_", "pp_"):
                if store_key.startswith(prefix):
                    cfg_key = store_key[len(prefix):]
                    # Handle pp_ sub-prefix like "pp_interpolate_interpolate_max_gap"
                    for pp_name in _POSTPROCESSOR_ORDER:
                        pfx2 = f"pp_{pp_name}_"
                        if store_key.startswith(pfx2):
                            cfg_key = store_key[len(pfx2):]
                            break
                    break
            else:
                cfg_key = store_key

            if isinstance(widget, QComboBox):
                config[cfg_key] = widget.currentText()
            elif isinstance(widget, QDoubleSpinBox):
                val = widget.value()
                config[cfg_key] = None if widget.specialValueText() == "none" and val == widget.minimum() else val
            else:
                config[cfg_key] = widget.value()

        # morph_kernel_size is stored as a single int in the widget; expand to [n, n]
        if "morph_kernel_size" in config and isinstance(config["morph_kernel_size"], int):
            config["morph_kernel_size"] = [config["morph_kernel_size"], config["morph_kernel_size"]]

        return config

    def apply_config(self, config):
        """Push a config dict into the panel widgets."""
        def _set_combo(combo, key):
            val = config.get(key)
            if val is not None:
                idx = combo.findText(str(val))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        _set_combo(self._combo_pre,  "preprocessor")
        _set_combo(self._combo_perc, "perception")
        _set_combo(self._combo_trk,  "tracker")

        if "fps" in config:
            self._spin_fps.setValue(float(config["fps"]))
        if "pixel_size" in config:
            self._spin_px.setValue(float(config["pixel_size"]))

        for pp_name, chk in self._pp_checkboxes.items():
            chk.setChecked(pp_name in config.get("postprocessors", []))

        # Push individual param values
        for store_key, widget in self._param_widgets.items():
            # Determine the bare config key by stripping whichever prefix matches
            cfg_key = store_key
            for pp_name in _POSTPROCESSOR_ORDER:
                if store_key.startswith(f"pp_{pp_name}_"):
                    cfg_key = store_key[len(f"pp_{pp_name}_"):]
                    break
            else:
                for prefix in ("pre_", "perc_", "trk_"):
                    if store_key.startswith(prefix):
                        cfg_key = store_key[len(prefix):]
                        break

            if cfg_key not in config:
                continue
            val = config[cfg_key]

            # morph_kernel_size stored as [n,n]; read first element
            if isinstance(val, list):
                val = val[0]

            if isinstance(widget, QComboBox):
                idx = widget.findText(str(val))
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                if val is not None:
                    widget.setValue(val)

    # ── file dialogs ──────────────────────────────────────────────────────────

    def _pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "",
            "Video files (*.mp4 *.avi *.mov *.mkv);;All files (*)"
        )
        if path:
            self._btn_run.setEnabled(True)
            self.video_requested.emit(path)

    def _pick_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Config", "configs/",
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            with open(path) as f:
                config = json.load(f)
            self.apply_config(config)
            self.config_loaded.emit(config)
        except Exception as e:
            QMessageBox.critical(self, "Config error", str(e))

    def _save_config(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config", "configs/",
            "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            config = self.get_config()
            with open(path, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _emit_run(self):
        self.run_requested.emit(self.get_config())

    # ── external enable/disable ───────────────────────────────────────────────

    def set_video_loaded(self, loaded: bool):
        self._btn_run.setEnabled(loaded)
        self._btn_preview_pre.setEnabled(loaded)
        self._btn_preview_det.setEnabled(loaded)

    def set_running(self, running: bool):
        """Disable all run buttons while pipeline is active."""
        for btn in (self._btn_run, self._btn_preview_pre, self._btn_preview_det):
            btn.setEnabled(not running)
