# YOLO Heatmap Fork

This fork adds a custom `predict` heatmap mode to Ultralytics so you can overlay model attention on top of normal prediction outputs.

## What is custom in this fork

- Adds `HeatMap=True` (alias of `heatmap=True`) to enable heatmap overlay during `model.predict(...)`.
- Adds `HMO=<float>` (alias of `hmo`) to control heatmap opacity.
- Keeps normal detections/segmentations/poses/obb renderings and overlays the heatmap on top.
- Includes compatibility testing script across available pretrained checkpoints.

## Install this fork locally

```bash
conda create -n yolo-heatmap python=3.11 -y
conda activate yolo-heatmap
cd /path/to/this/repo
pip install -e .
```

> This feature is fork-specific. It is not available in the default PyPI package unless you install this modified source tree.

## Quick usage

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# Baseline output
model.predict(
    "https://ultralytics.com/images/bus.jpg",
    save=True,
    imgsz=320,
    conf=0.25,
    project="runs/detect",
    name="baseline",
    exist_ok=True,
)

# Baseline + heatmap overlay
model.predict(
    "https://ultralytics.com/images/bus.jpg",
    save=True,
    imgsz=320,
    conf=0.25,
    HeatMap=True,  # also supports: heatmap=True
    HMO=0.8,       # also supports: hmo=0.8
    project="runs/detect",
    name="heatmap",
    exist_ok=True,
)
```

Expected output images:

- `runs/detect/baseline/bus.jpg`
- `runs/detect/heatmap/bus.jpg`

## Compatibility test (this fork)

Script: `test_heatmap_all_models.py`

Run:

```bash
python test_heatmap_all_models.py
```

Default settings:

- Versions: `8 9 11 12 26`
- Tasks: `detect segment pose obb`
- Sizes: `auto` (version/task-aware)
- Classification: excluded by default

The script only tests valid pretrained checkpoint combinations for each version/task.

Latest local result:

- **71/71 passed**
- **0 failed**

## Availability notes used by the script

- YOLOv8: detect/segment/pose/obb (`n,s,m,l,x`)
- YOLOv9: detect (`t,s,m,c,e`) + segment (`c` only)
- YOLO11: detect/segment/pose/obb (`n,s,m,l,x`)
- YOLO12: detect only (`n,s,m,l,x`)
- YOLO26: detect/segment/pose/obb (`n,s,m,l,x`)

