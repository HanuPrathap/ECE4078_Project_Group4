# train_yolov8_local.py
import os
import sys
import pickle
from pathlib import Path

# --- Paths on Desktop ---
DESKTOP = Path.home() / "Desktop"
PROJECT = DESKTOP / "ECE4078_2023" / "groceries_detector_yolov8"
DATASETS = PROJECT / "datasets"
RUNS_OUT = PROJECT / "runs"

for p in [PROJECT, DATASETS, RUNS_OUT]:
    p.mkdir(parents=True, exist_ok=True)

print(f"Project dir: {PROJECT}")
print(f"Datasets dir: {DATASETS}")
print(f"Runs dir:     {RUNS_OUT}")

# --- Imports and checks ---
try:
    import ultralytics
    from ultralytics import YOLO
except Exception:
    raise SystemExit("Ultralytics missing. Run: pip install ultralytics==8.0.134")

print(f"Ultralytics version: {ultralytics.__version__}")

try:
    import torch
except Exception:
    raise SystemExit("PyTorch missing. Run: pip install torch torchvision")

from roboflow import Roboflow

# --- Device selection ---
if torch.cuda.is_available():
    device_sel = 0
    try:
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        gpu_name = "CUDA device 0"
    print(f"Using GPU: {gpu_name}")
else:
    device_sel = "cpu"
    print("Using CPU")

# --- PyTorch 2.6 safe load allowlist for Ultralytics models ---
# This avoids _pickle.UnpicklingError when loading pretrained *.pt files
try:
    from torch.serialization import add_safe_globals
    from ultralytics.nn.tasks import DetectionModel
    add_safe_globals([DetectionModel])
    print("Registered Ultralytics DetectionModel in torch safe globals.")
except Exception as e:
    print(f"Safe globals registration skipped or not needed: {e}")

# --- Roboflow dataset download (your updated project) ---
API_KEY = "ivXFj8FxjhcdDz4IPvK4"
WORKSPACE = "ece4078g04"
PROJECT_NAME = "veg-classifier-4-pzuuk"
VERSION = 14
FORMAT = "yolov8"

print("Connecting to Roboflow...")
rf = Roboflow(api_key=API_KEY)
rf_project = rf.workspace(WORKSPACE).project(PROJECT_NAME)

# Download under DATASETS
os.chdir(DATASETS)
version = rf_project.version(VERSION)
dataset = version.download(FORMAT)
print(f"Roboflow dataset at: {dataset.location}")

# --- Prepare training ---
data_yaml = Path(dataset.location) / "data.yaml"
if not data_yaml.exists():
    raise FileNotFoundError(f"Could not find data.yaml at {data_yaml}")

# Change to project folder so outputs land on Desktop
os.chdir(PROJECT)

# Choose model
# Tip: use "yolov8n.pt" for faster CPU training, "yolov8s.pt" for better accuracy
model_name = "yolov8s.pt"

# Try to build model with pretrained weights, if that fails fall back to yaml
def build_model(model_spec: str) -> YOLO:
    try:
        return YOLO(model_spec)
    except pickle.UnpicklingError as e:
        print("Caught UnpicklingError while loading weights. "
              "Falling back to YAML (training from scratch).")
        base = Path(model_spec).stem
        yaml_name = f"{base}.yaml" if model_spec.endswith(".pt") else model_spec
        return YOLO(yaml_name)

model = build_model(model_name)

# --- Train ---
print("Starting training...")
train_results = model.train(
    data=str(data_yaml),
    epochs=100,
    imgsz=320,
    batch=12,
    device=device_sel,
    project=str(RUNS_OUT),
    name="detect_train",
    verbose=True,
)

print("\nTraining complete.")
try:
    print(f"Best weights: {train_results.best}")
except Exception:
    pass

# --- Validate ---
print("Running validation...")
model.val(
    data=str(data_yaml),
    imgsz=320,
    device=device_sel,
    project=str(RUNS_OUT),
    name="detect_val",
)

# --- Predict on test images if present ---
test_images = Path(dataset.location) / "test" / "images"
if test_images.exists() and any(test_images.iterdir()):
    print("Running prediction on test images...")
    model.predict(
        source=str(test_images),
        conf=0.25,
        imgsz=320,
        save=True,
        project=str(RUNS_OUT),
        name="detect_predict",
        device=device_sel,
    )
    print(f"Predictions saved to: {RUNS_OUT / 'detect_predict'}")
else:
    print(f"No test images found at {test_images}. Skipping prediction.")

print("Done.")
