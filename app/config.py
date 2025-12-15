import os

# 1. Get the directory where THIS file (config.py) is located (app/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the project root, then into 'model'
# This works regardless of which user or computer runs the code
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best.onnx")

# 3. Clean up the path (removes the "..")
MODEL_PATH = os.path.normpath(MODEL_PATH)

CONFIDENCE_THRESHOLD = 0.4