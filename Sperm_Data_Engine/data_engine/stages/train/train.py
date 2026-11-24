"""
train.py -- unified YOLO training / continuation CLI for the data engine.

Replaces train_2.py + fine-tune.py: one script, `--resume-from` switches it
from cold-start (from --base-weights, default yolo11x.pt) into continuing an
existing checkpoint. Reads the dataset from the registry
(data_engine/registry/manifest.yaml) instead of a hand-edited yaml file.

Carries forward two fixes discovered the hard way this session:
  - workers=2 by default, not the original scripts' workers=0, which let the
    main process's memory grow every epoch until Windows started swapping to
    disk (confirmed via Task Manager: 30GB+ committed memory against 32GB
    total RAM; epochs went from ~9min to 30-75min as a result).
  - plots=False by default, a suspected contributor to the same leak
    (Ultralytics regenerates several matplotlib figures every epoch).

Output location: rather than fighting Ultralytics' project=/name= path
nesting (observed this session to land under runs/detect/<project>/<name>/
instead of <project>/<name>/ directly), this just asks the trained model
where it actually put things afterward (model.trainer.save_dir) and writes
run_manifest.json there -- robust regardless of Ultralytics version quirks.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.train.train --epochs 1 --tiers 10
    python -m data_engine.stages.train.train --resume-from training_weights/experiment_5/weights/best.pt --epochs 20 --lr0 0.001
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # prevents the OMP Error #15 crash

import torch
import yaml
from ultralytics import YOLO

from data_engine.registry.registry import Registry
from data_engine.registry.to_yolo_yaml import build_yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BASE_WEIGHTS = REPO_ROOT / "yolo11x.pt"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def _make_progress_callback(progress_file: Path):
    """Ultralytics on_train_epoch_end callback writing {epoch, epochs,
    save_dir, train_loss, timestamp} as JSON to `progress_file` after every
    epoch -- for the GUI hub's Train tab to poll while this runs as a
    detached subprocess. `progress_file`'s path is dictated by the caller
    (the GUI, under its own job directory) rather than guessed from
    Ultralytics' project=/name= path nesting -- the exact fragility this
    module's own docstring already flags for `save_dir` resolution, so no
    reason to reintroduce it here just to find where to write progress."""
    def _on_train_epoch_end(trainer):
        try:
            train_loss = trainer.label_loss_items(trainer.tloss, prefix="train")
        except Exception:
            train_loss = None
        progress_file.parent.mkdir(parents=True, exist_ok=True)
        progress_file.write_text(json.dumps({
            "epoch": trainer.epoch + 1,
            "epochs": trainer.epochs,
            "save_dir": str(trainer.save_dir),
            "train_loss": train_loss,
            "timestamp": datetime.now().isoformat(),
        }))
    return _on_train_epoch_end


def train(
    base_weights=None,
    resume_from=None,
    tiers=None,
    epochs=50,
    imgsz=1024,
    batch=8,
    lr0=0.01,
    workers=2,
    plots=False,
    device=None,
    notes="",
    progress_file=None,
):
    if resume_from and base_weights:
        raise ValueError("Pass only one of base_weights / resume_from")
    weights_path = Path(resume_from) if resume_from else Path(base_weights or DEFAULT_BASE_WEIGHTS)

    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    registry = Registry.load()
    dataset_doc = build_yaml(registry, tiers=tiers)
    if not dataset_doc["train"] or not dataset_doc["val"]:
        raise ValueError(
            f"No train/val clips matched tiers={tiers}. "
            f"Have {len(dataset_doc['train'])} train, {len(dataset_doc['val'])} val clips."
        )

    clip_ids = sorted(
        c.clip_id for c in registry.clips
        if tiers is None or c.density_tier in tiers
    )

    run_id = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    print(f"--- {run_id} ---")
    print(f"Weights: {'resume ' if resume_from else 'cold-start '}{weights_path}")
    print(f"Device: {device}  |  clips: {len(clip_ids)} "
          f"({len(dataset_doc['train'])} train / {len(dataset_doc['val'])} val)")

    if resume_from and epochs == 50 and lr0 == 0.01:
        print(
            "NOTE: --resume-from was given with default epochs/lr0 unchanged. True "
            "fine-tuning conventionally uses fewer epochs and a lower learning rate than "
            "cold-start training (e.g. --epochs 20 --lr0 0.001), not more epochs at an "
            "unchanged rate."
        )

    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml.safe_dump(dataset_doc, tmp)
    tmp.close()
    dataset_yaml_path = tmp.name

    model = YOLO(str(weights_path))
    if progress_file is not None:
        model.add_callback("on_train_epoch_end", _make_progress_callback(Path(progress_file)))

    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=device,
        workers=workers,
        plots=plots,
        project=str(MODELS_DIR),
        name=run_id,
        exist_ok=False,
    )
    save_dir = Path(model.trainer.save_dir)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "git_commit": _git_commit(),
        "base_weights": None if resume_from else str(weights_path),
        "resumed_from": str(weights_path) if resume_from else None,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "lr0": lr0,
        "workers": workers,
        "plots": plots,
        "device": device,
        "tiers": tiers,
        "clip_ids": clip_ids,
        "n_train_clips": len(dataset_doc["train"]),
        "n_val_clips": len(dataset_doc["val"]),
        "notes": notes,
        "save_dir": str(save_dir),
    }
    with open(save_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    shutil.copy(dataset_yaml_path, save_dir / "dataset_used.yaml")
    os.remove(dataset_yaml_path)

    print(f"\nDone. Weights + run_manifest.json in: {save_dir}")
    return save_dir


def build_parser() -> argparse.ArgumentParser:
    """Split out from _cli() so the GUI hub's Train tab can build its form
    from the exact same flag definitions (and, since that tab has to
    produce a subprocess argv list either way, reflection here is the
    simpler option rather than an added layer -- unlike Prepare Data/
    Evaluate, which call their stage functions directly and hand-write
    forms against those instead)."""
    parser = argparse.ArgumentParser(description="Train or continue a YOLO sperm-detection model")
    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument("--base-weights", default=None,
                                help=f"Cold-start weights (default: {DEFAULT_BASE_WEIGHTS})")
    weights_group.add_argument("--resume-from", default=None,
                                help="Continue training from this checkpoint instead of cold-starting")
    parser.add_argument("--tiers", action="append", default=None,
                         help="Filter registry clips to this density tier (repeatable; default: all)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plots", action="store_true",
                         help="Enable Ultralytics' per-epoch diagnostic plots (default: off)")
    parser.add_argument("--device", default=None, help="Default: cuda:0 if available, else cpu")
    parser.add_argument("--notes", default="")
    parser.add_argument("--progress-file", default=None,
                         help="Write JSON training progress here after every epoch (for the GUI "
                              "hub's Train tab to poll; not needed for plain CLI use)")
    return parser


def _cli():
    args = build_parser().parse_args()

    train(
        base_weights=args.base_weights,
        resume_from=args.resume_from,
        tiers=args.tiers,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        workers=args.workers,
        plots=args.plots,
        device=args.device,
        notes=args.notes,
        progress_file=args.progress_file,
    )


if __name__ == "__main__":
    _cli()
