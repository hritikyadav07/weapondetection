import os
import sys

# Ensure ultralytics is available
try:
    from ultralytics import YOLO
except ImportError:
    print("Ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)

# Config (defaults tuned for GTX 1650 4GB)
DATA_YAML = os.environ.get('YOLO_DATA') or os.path.join(os.path.dirname(__file__), 'data.yaml')
MODEL = os.environ.get('YOLO_MODEL', 'yolov8s.pt')  # small model fits 4GB well
IMG_SIZE = int(os.environ.get('IMG', 640))
EPOCHS = int(os.environ.get('EPOCHS', 100))
BATCH = int(os.environ.get('BATCH', -1))  # -1 lets YOLO auto-find batch size
WORKERS = int(os.environ.get('WORKERS', 2))  # fewer workers for Windows stability
PATIENCE = int(os.environ.get('PATIENCE', 30))
PROJECT = os.environ.get('PROJECT', 'runs')
NAME = os.environ.get('NAME', 'weapons-yolov8s')
DEVICE = os.environ.get('DEVICE', '0')  # 'cpu' or CUDA device index like '0'
SEED = int(os.environ.get('SEED', 42))


def main():
    print(f"Training with data={DATA_YAML}, model={MODEL}")
    model = YOLO(MODEL)
    results = model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH,
        workers=WORKERS,
        patience=PATIENCE,
        project=PROJECT,
        name=NAME,
        device=DEVICE,
        seed=SEED,
    amp=False,  # disable AMP to avoid potential NaNs on some GPUs/driver combos
    cos_lr=False,
        plots=False,
        # safer augments to prevent extreme boxes/NaNs
    mixup=0.0,
    mosaic=0.0,  # disable mosaic for stability
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.25,
    degrees=0.0,
    translate=0.05,
    scale=0.2,
    shear=0.0,
    perspective=0.0,
    optimizer='SGD',
    lr0=0.001,
    lrf=0.01,
    )
    print(results)


if __name__ == '__main__':
    main()
