# Weapon Detection Training

This repo contains a YOLOv8 training setup for your weapon dataset exported from Roboflow.

## Quick start

1. Create a Python env and install dependencies.
2. Run a short smoke train to verify setup.
3. Start a long training for best accuracy.

### Setup

```bash
# create venv (Windows Git Bash)
python -m venv .venv
source .venv/Scripts/activate
pip install --upgrade pip
pip install ultralytics
```

### Smoke test (1 epoch)

```bash
python train_yolo.py \
  && echo "If this fails, check data.yaml paths and label files."
```

### Full training (recommended)

```bash
# Examples of accuracy-first configs; adjust to your GPU memory
EPOCHS=200 BATCH=16 IMG=640 DEVICE=0 NAME=weapons-yolov8m python train_yolo.py
# Larger models for higher accuracy (need more VRAM)
EPOCHS=200 BATCH=8 IMG=640 DEVICE=0 YOLO_MODEL=yolov8l.pt NAME=weapons-yolov8l python train_yolo.py
EPOCHS=300 BATCH=6 IMG=640 DEVICE=0 YOLO_MODEL=yolov8x.pt NAME=weapons-yolov8x python train_yolo.py
```

Artifacts are saved under `runs/detect/<NAME>`.

## Tips for higher accuracy

- Try a larger model (`yolov8l.pt` or `yolov8x.pt`).
- Train longer with `EPOCHS=300` and enable cosine LR (already on) and strong augmentations (configured in script).
- Increase image size (e.g., `IMG=800`) if GPU allows.
- Ensure class names in `data.yaml` are correct and meaningful. Current classes appear as: `['-', 'gun', 'gun-knife-thesis - v11 Yolov5 augmented', 'knife']`.
- Clean duplicate or noisy labels; dataset looks to have multiple label files per image in some folders—keep exactly one label file per image.

## Inference after training

```bash
# Replace the path below with your best weights file
python - <<'PY'
from ultralytics import YOLO
m = YOLO('runs/weapons-yolov8m/weights/best.pt')
m.predict(source='test/images', imgsz=640, conf=0.25, device='0')
PY
```

