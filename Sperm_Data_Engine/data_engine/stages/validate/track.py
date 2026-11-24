"""
track.py -- run a trained YOLO model + ByteTrack/BoT-SORT over one video.

Same core loop as track_video_bytetrack.py, minus the tkinter file dialogs
and the live cv2.imshow preview (not useful in a CLI/automation context).
Same output schema: frame, track_id, class_id, x, y, w, h, conf.

track_video_botsort.py in the reference code is identical to
track_video_bytetrack.py except for the tracker config name, so both are
supported here via the same code path. track_video_ocsort.py depends on the
separate `boxmot` package (not installed, not used by any proven run this
session) and isn't supported here.

Exposes run_tracking() as a plain function so evaluate.py calls it directly
instead of shelling out.

Run as a module from the repo root, e.g.:

    python -m data_engine.stages.validate.track --model best.pt --video data/40_4/vid4.mp4 --out preds.csv
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # prevents the OMP Error #15 crash

import cv2
import pandas as pd
from ultralytics import YOLO

TRACKER_YAMLS = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}


def run_tracking(model_path, video_path, tracker: str = "bytetrack", conf: float = 0.40) -> pd.DataFrame:
    if tracker not in TRACKER_YAMLS:
        raise ValueError(f"Unknown tracker {tracker!r}; choose from {list(TRACKER_YAMLS)}")

    model = YOLO(str(model_path))
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    rows = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, tracker=TRACKER_YAMLS[tracker], conf=conf, verbose=False
        )
        result = results[0]

        if result.boxes.id is not None:
            boxes = result.boxes.xywh.cpu().numpy()
            track_ids = result.boxes.id.int().cpu().numpy()
            cls_ids = result.boxes.cls.int().cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for (x, y, w, h), track_id, cls_id, c in zip(boxes, track_ids, cls_ids, confs):
                rows.append([frame_idx, int(track_id), int(cls_id),
                             float(x), float(y), float(w), float(h), float(c)])

        frame_idx += 1

    cap.release()
    return pd.DataFrame(
        rows, columns=["frame", "track_id", "class_id", "x", "y", "w", "h", "conf"]
    )


def _cli():
    parser = argparse.ArgumentParser(description="Run YOLO + ByteTrack/BoT-SORT on a video")
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracker", choices=list(TRACKER_YAMLS), default="bytetrack")
    parser.add_argument("--conf", type=float, default=0.40)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = run_tracking(args.model, args.video, tracker=args.tracker, conf=args.conf)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    _cli()
