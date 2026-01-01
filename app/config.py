import os

# 1. Get the directory where THIS file (config.py) is located (app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Define paths for BOTH models
# Old v9 Model
MODEL_PATH_V9 = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "yolov9.onnx"))

# --- YOLOv8 (New Model - SWITCH TO .PT) ---
# We are changing this from .onnx to .pt to fix the version error
MODEL_PATH_V8 = os.path.normpath(os.path.join(BASE_DIR, "..", "model", "yolov8n.pt"))
# 3. Thresholds
CONFIDENCE_THRESHOLD = 0.35