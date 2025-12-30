import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import easyocr
from collections import Counter
import re
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD
from ultralytics import YOLO

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
        Context-Aware Logic for Sri Lankan Plates:
        1. Zoom & Enhance.
        2. Detect Layout (Wide vs Square).
        3. Sort text based on layout (Vertical priority for Square).
        4. Apply "Cheat Codes" (Valid Prefixes & Number formats).
        """
        candidates = []
        
        # The 9 Valid Provinces
        VALID_PROVINCES = ["WP", "EP", "NP", "SP", "SG", "NC", "CP", "UP", "NW"]

        def fix_province_code(text):
            """Fixes common OCR errors in province codes (e.g., '5P' -> 'SP')"""
            text = text.upper().replace(" ", "")
            # Direct matches
            for prov in VALID_PROVINCES:
                if prov in text:
                    return text # It's already correct
            
            # Fuzzy fixes
            fixes = {
                "5P": "SP", "8P": "SP", "S9": "SP",
                "W9": "WP", "VP": "WP", "VV": "WP",
                "NWC": "NW", "HCP": "CP", "C9": "CP"
            }
            for wrong, right in fixes.items():
                if wrong in text:
                    return text.replace(wrong, right)
            return text

        def get_structured_text(image_input):
            # 1. ZOOM IN
            w, h = image_input.size
            image_input = image_input.resize((w * 2, h * 2), Image.LANCZOS)
            img_array = np.array(image_input)

            # 2. READ EVERYTHING WITH COORDINATES
            results = self.reader.readtext(img_array, detail=1)
            if not results: return ""
            
            # Filter low confidence
            results = [r for r in results if r[2] > 0.35]
            if not results: return ""

            # 3. DETECT LAYOUT
            # Get Y-centers of all text blocks
            y_centers = [(r[0][0][1] + r[0][2][1]) / 2 for r in results]
            height_span = max(y_centers) - min(y_centers)
            img_h = img_array.shape[0]
            
            # If text is spread out vertically (>30% of image height), it's STACKED
            is_stacked = height_span > (img_h * 0.3)

            final_text_parts = []

            if is_stacked:
                # --- SQUARE PLATE LOGIC (Your "Top-to-Bottom" Idea) ---
                # Sort primarily by Y (Vertical), secondary by X (Horizontal)
                # This ensures "SP QL" (Top) comes before "9904" (Bottom)
                results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                
                # Split into Top and Bottom rows for strict cleaning
                mid_y = img_h / 2
                top_text = []
                bottom_text = []
                
                for r in results:
                    y_c = (r[0][0][1] + r[0][2][1]) / 2
                    text_val = r[1].upper().replace(".", "").replace("-", "")
                    
                    if y_c < mid_y:
                        # TOP ROW: Should contain Province (SP) and Class (QL)
                        # Fix numbers that look like letters here if needed
                        top_text.append(text_val)
                    else:
                        # BOTTOM ROW: Should be Numbers (9904)
                        # Force 'O'->'0', 'I'->'1', 'B'->'8'
                        text_val = text_val.replace("O", "0").replace("I", "1").replace("B", "8")
                        bottom_text.append(text_val)
                
                # Join them
                full_top = "".join(top_text)
                full_bottom = "".join(bottom_text)
                
                # Apply Prefix Correction (e.g., fix "5P" -> "SP")
                full_top = fix_province_code(full_top)
                
                final_text = full_top + full_bottom

            else:
                # --- WIDE PLATE LOGIC (Left-to-Right) ---
                results.sort(key=lambda r: r[0][0][0])
                raw_text = "".join([r[1] for r in results]).upper()
                
                # Clean up common separators
                final_text = raw_text.replace("-", "").replace(" ", "").replace(".", "")
                
                # Apply Prefix Correction
                final_text = fix_province_code(final_text)

            return final_text

        # --- RUN VARIATIONS ---
        candidates.append(get_structured_text(plate_crop))
        
        gray = ImageOps.grayscale(plate_crop)
        enhancer = ImageEnhance.Contrast(gray)
        high_contrast = enhancer.enhance(2.0)
        candidates.append(get_structured_text(high_contrast))

        # --- VOTING ---
        cleaned = [self.clean_text(c) for c in candidates]
        cleaned = [c for c in cleaned if len(c) > 3]

        if not cleaned: return "Unknown"
        return Counter(cleaned).most_common(1)[0][0]




        #end of easyocr

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
            
            # --- ADD PADDING (Crucial for edge text like 'WP') ---
            pad = 10  # Add 10 pixels of breathing room
            crop_x1 = max(0, x1 - pad)
            crop_y1 = max(0, y1 - pad)
            crop_x2 = min(orig_w, x2 + pad)
            crop_y2 = min(orig_h, y2 + pad)

            # Crop the plate with padding
            plate_crop = self.original_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
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