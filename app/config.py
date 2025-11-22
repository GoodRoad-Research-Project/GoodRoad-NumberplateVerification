import os

# Configuration settings
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/kanishka/Goodroad/IT22174444/license_plate_detection_service/model/best.onnx")
CONFIDENCE_THRESHOLD = 0.4