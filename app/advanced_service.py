import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import easyocr
from collections import Counter
import re
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD

class AdvancedYOLOService:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)
        
        # --- ENSEMBLE SETUP ---
        # We initialize EasyOCR once. 
        # 'gpu=False' is safer for Mac. If you have a strong NVIDIA GPU, set True.
        print("🚀 Initializing Ensemble OCR Engine...")
        self.reader = easyocr.Reader(['en'], gpu=False) 
        print("✅ Advanced OCR Ready")

    def load_model(self):
        """Loads the ONNX model."""
        try:
            print(f"🚀 Loading Model from {MODEL_PATH}...")
            self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]
            print("✅ Model loaded!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.session = None

    def preprocess(self, image_bytes: bytes):
        """Standard resize and normalize."""
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        self.original_image = img # Keep original for cropping
        original_size = img.size
        
        img_resized = img.resize(self.input_shape)
        img_data = np.array(img_resized).astype(np.float32)
        img_data /= 255.0
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data, original_size

    def clean_text(self, text):
        """Removes special characters to just get the plate number."""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def ensemble_ocr(self, plate_crop):
        """
        Runs OCR on 3 variations of the image and votes on the best result.
        """
        candidates = []

        # 1. Original
        res1 = self.reader.readtext(np.array(plate_crop), detail=0)
        candidates.extend(res1)

        # 2. High Contrast Grayscale
        gray = ImageOps.grayscale(plate_crop)
        enhancer = ImageEnhance.Contrast(gray)
        high_contrast = enhancer.enhance(2.0)
        res2 = self.reader.readtext(np.array(high_contrast), detail=0)
        candidates.extend(res2)

        # 3. Binarized (Black and White only)
        # Threshold: Pixels brighter than 128 become white, others black
        arr = np.array(gray)
        binary = np.where(arr > 128, 255, 0).astype(np.uint8)
        res3 = self.reader.readtext(binary, detail=0)
        candidates.extend(res3)

        # --- VOTING LOGIC ---
        if not candidates:
            return "Unknown"

        # Clean all results (remove garbage chars)
        cleaned_candidates = [self.clean_text(c) for c in candidates]
        # Remove empty strings
        cleaned_candidates = [c for c in cleaned_candidates if len(c) > 1]

        if not cleaned_candidates:
            return "Unknown"

        # Count the votes
        vote_counts = Counter(cleaned_candidates)
        # Get the winner (most common result)
        most_common_text, count = vote_counts.most_common(1)[0]
        
        return most_common_text

    def predict(self, image_bytes: bytes) -> list:
        if not self.session: raise RuntimeError("Model not loaded")

        input_tensor, (orig_w, orig_h) = self.preprocess(image_bytes)
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        predictions = np.squeeze(outputs[0]).T

        # Filter by confidence
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > CONFIDENCE_THRESHOLD, :]
        scores = scores[scores > CONFIDENCE_THRESHOLD]

        if len(predictions) == 0: return []

        # NMS & Scaling
        boxes = predictions[:, :4]
        indices = self.nms(boxes, scores, 0.45)

        x_scale = orig_w / self.input_shape[0]
        y_scale = orig_h / self.input_shape[1]
        
        results = []
        for i in indices:
            cx, cy, w, h = boxes[i]
            x1 = int((cx - w/2) * x_scale)
            y1 = int((cy - h/2) * y_scale)
            x2 = int((cx + w/2) * x_scale)
            y2 = int((cy + h/2) * y_scale)
            
            # Clamp coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            # --- OCR EXECUTION ---
            # Crop the plate
            plate_crop = self.original_image.crop((x1, y1, x2, y2))
            # Run the ensemble
            plate_text = self.ensemble_ocr(plate_crop)

            results.append({
                "class_name": "license_plate",
                "text": plate_text,
                "confidence": round(float(scores[i]), 2),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            })
        
        return results

    def draw_detections(self, image_bytes: bytes, detections: list) -> bytes:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        
        for det in detections:
            b = det["bbox"]
            text = det.get("text", "")
            draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="lime", width=4)
            # Draw text
            if text:
                draw.rectangle([b["x1"], b["y1"]-25, b["x1"]+150, b["y1"]], fill="lime")
                draw.text((b["x1"]+5, b["y1"]-20), f"{text} ({det['confidence']})", fill="black")
                
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

    def nms(self, boxes, scores, thresh):
        x1 = boxes[:, 0] - boxes[:, 2]/2
        y1 = boxes[:, 1] - boxes[:, 3]/2
        x2 = boxes[:, 0] + boxes[:, 2]/2
        y2 = boxes[:, 1] + boxes[:, 3]/2
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2-xx1)
            h = np.maximum(0.0, yy2-yy1)
            inter = w*h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

# Export the instance
detector = AdvancedYOLOService()