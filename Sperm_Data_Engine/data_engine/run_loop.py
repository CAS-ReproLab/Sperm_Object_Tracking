"""
run_loop.py -- chains training straight into evaluation.

The concrete pain point this removes: train.py prints a timestamped run
path (data_engine/models/run_<timestamp>/) that you'd otherwise have to
copy by hand into evaluate.py's --model argument. This calls both stages'
already-plain-importable functions directly (train.train(), then
evaluate.evaluate() against exactly the checkpoint that just came out of
it) and prints both result sets together.

Scope, deliberately: one train + one evaluate. Labeling stays a separate,
manual/interactive step (stages/label/) -- it can't be chained in the same
way since it blocks on a human. Chaining multiple rounds together, or
folding labeling in too, is future scope if wanted.

Run as a module from the repo root, e.g.:

    python -m data_engine.run_loop --tiers 40 --epochs 50 --eval-tiers 40
    python -m data_engine.run_loop --resume-from training_weights/experiment_5/weights/best.pt --epochs 20 --lr0 0.001
"""
import argparse

from data_engine.stages.train.train import train
from data_engine.stages.validate.evaluate import evaluate


def run_loop(
    base_weights=None, resume_from=None, tiers=None, epochs=50, imgsz=1024,
    batch=8, lr0=0.01, workers=2, plots=False, device=None, notes="",
    eval_tiers=None, eval_tracker="bytetrack",
):
    print("=== Training ===")
    save_dir = train(
        base_weights=base_weights, resume_from=resume_from, tiers=tiers,
        epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, workers=workers,
        plots=plots, device=device, notes=notes,
    )

    model_path = save_dir / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Training finished but {model_path} wasn't found -- check {save_dir}")

    print(f"\n=== Evaluating {model_path} ===")
    eval_out_dir = evaluate(model_path, tiers=eval_tiers, tracker=eval_tracker)

    print(f"\nDone. Weights: {model_path}\nEvaluation: {eval_out_dir}")
    return model_path, eval_out_dir


def _cli():
    parser = argparse.ArgumentParser(description="Train a model, then evaluate it -- no manual path copying")

    weights_group = parser.add_mutually_exclusive_group()
    weights_group.add_argument("--base-weights", default=None)
    weights_group.add_argument("--resume-from", default=None)
    parser.add_argument("--tiers", action="append", default=None,
                         help="Filter training clips to this density tier (repeatable; default: all)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--notes", default="")

    parser.add_argument("--eval-tiers", action="append", default=None,
                         help="Filter val clips for evaluation to this tier (repeatable; default: all)")
    parser.add_argument("--eval-tracker", choices=["bytetrack", "botsort"], default="bytetrack")

    args = parser.parse_args()

    run_loop(
        base_weights=args.base_weights, resume_from=args.resume_from, tiers=args.tiers,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, lr0=args.lr0,
        workers=args.workers, plots=args.plots, device=args.device, notes=args.notes,
        eval_tiers=args.eval_tiers, eval_tracker=args.eval_tracker,
    )


if __name__ == "__main__":
    _cli()
