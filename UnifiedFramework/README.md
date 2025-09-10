# Unified Framework
This repository contains code for sperm object tracking in microscopy videos.This is the official code for [A framework for evaluating predicted sperm trajectories in crowded microscopy videos](https://www.biorxiv.org/content/10.1101/2025.02.20.639342v1).
It runs on the [Trackpy Libary](https://soft-matter.github.io/trackpy/v0.7/). For teh tracking metrics, the code is taken from [Py-CTCMetrics](https://github.com/CellTrackingChallenge/py-ctcmetrics).

To get started, once Trackpy, OpenCV, and other needed python packages are installed, simply run

`python main.py`

and select the video file and recommended configuration file for your video. We include three example video files of 10X magnification in the SchimdtDataset folder.

Alternatively, if you want to use more direct modification, you can run the `sandbox_template.ipynb` Jupyter notebook file.

<p align="center">
  <img src="assets/tracking_example.png" />
</p>


## Understanding the Code

The code base is broken into 5 primary files.
- `tracker.py`: Runs the Trackpy detector and linking algorithm and outputs a .csv file of each sperm id in each frame.
- `stats.py`: Computes CASA motility parameters for each sperm.
- `visualizer.py`: Outputs videos of sperm trails, bounding boxes, and other visualizations.
- `labeler.py`: Can be used to hand correct videos to create ground truth labeled data.
- `metrics.py`: Takes prediction data and groudtruth data and runs detection and tracking metrics.

If you use the code in this repo, please cite:

```bibtex
@article {Hart2025framework,
	author = {Hart, David and Cashwell, Kylie and Bhandari, Anita and Schmidt, Cameron},
	title = {A framework for evaluating predicted sperm trajectories in crowded microscopy videos},
	elocation-id = {2025.02.20.639342},
	year = {2025},
	doi = {10.1101/2025.02.20.639342},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2025/02/26/2025.02.20.639342},
	eprint = {https://www.biorxiv.org/content/early/2025/02/26/2025.02.20.639342.full.pdf},
	journal = {bioRxiv}
}
```



