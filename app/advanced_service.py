import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import easyocr
from collections import Counter
from ultralytics import YOLO
from app.config import MODEL_PATH_V9, MODEL_PATH_V8, CONFIDENCE_THRESHOLD

class AdvancedYOLOService:
    def __init__(self):
        self.session_v9 = None
        self.model_v8 = None
        self.input_name_v9 = None
        self.output_names_v9 = None
        
        print("🚀 Initializing Hybrid Ensemble Engine...")
        
        # 1. Initialize EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
        # 2. Load Both Models
        self.load_models()

    def load_models(self):
        """Loads YOLOv9 (ONNX) and YOLOv8 (PT)"""
        # --- Load YOLOv9 (ONNX) ---
        try:
            print(f"🚀 Loading YOLOv9 from {MODEL_PATH_V9}...")
            self.session_v9 = ort.InferenceSession(MODEL_PATH_V9, providers=["CPUExecutionProvider"])
            self.input_name_v9 = self.session_v9.get_inputs()[0].name
            self.output_names_v9 = [o.name for o in self.session_v9.get_outputs()]
            print("✅ YOLOv9 Loaded")
        except Exception as e:
            print(f"❌ Error loading YOLOv9: {e}")

        # --- Load YOLOv8 (PT) ---
        try:
            print(f"🚀 Loading YOLOv8 from {MODEL_PATH_V8}...")
            # This works for both .pt and .onnx automatically
            self.model_v8 = YOLO(MODEL_PATH_V8, task='detect')
            print("✅ YOLOv8 Loaded")
        except Exception as e:
            print(f"❌ Error loading YOLOv8: {e}")

    def predict(self, image_bytes: bytes):
        """
        Runs Hybrid Inference: v9 + v8 -> NMS -> Smart OCR
        """
        # 1. Prepare Image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        # 2. Run Both Models
        boxes_v9 = self.run_yolov9_onnx(img)  # List of [x1, y1, x2, y2, score]
        boxes_v8 = self.run_yolov8(img)       # RENAMED: Generic V8 runner
        
        # 3. Combine & Fuse (Ensemble)
        all_boxes = boxes_v9 + boxes_v8
        
        if not all_boxes:
            return []

        # Convert to numpy for NMS
        all_boxes_np = np.array(all_boxes)
        
        # Run NMS (Non-Maximum Suppression)
        keep_indices = self.nms(
            boxes=all_boxes_np[:, :4], 
            scores=all_boxes_np[:, 4], 
            thresh=0.45
        )
        
        final_detections = []
        for i in keep_indices:
            bbox = all_boxes_np[i, :4].astype(int)
            conf = float(all_boxes_np[i, 4])
            
            # 4. Context-Aware OCR
            plate_crop = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            plate_text = self.ensemble_ocr(plate_crop)
            
            final_detections.append({
                "class_name": "License Plate",
                "confidence": round(conf, 2),
                "bbox": {
                    "x1": int(bbox[0]), "y1": int(bbox[1]), 
                    "x2": int(bbox[2]), "y2": int(bbox[3])
                },
                "text": plate_text
            })
            
        return final_detections

    # --- HELPER: Run YOLOv9 (Raw ONNX) ---
    def run_yolov9_onnx(self, img):
        if not self.session_v9: return []
        
        # Resize to 640x640
        img_resized = img.resize((640, 640))
        input_data = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        input_data = input_data[None, ...]
        
        outputs = self.session_v9.run(self.output_names_v9, {self.input_name_v9: input_data})
        predictions = np.squeeze(outputs[0]).T
        
        boxes_list = []
        img_w, img_h = img.size
        
        for pred in predictions:
            score = pred[4]
            if score > CONFIDENCE_THRESHOLD:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                x1 = (cx - w/2) * (img_w / 640)
                y1 = (cy - h/2) * (img_h / 640)
                x2 = (cx + w/2) * (img_w / 640)
                y2 = (cy + h/2) * (img_h / 640)
                boxes_list.append([x1, y1, x2, y2, score])
        return boxes_list

    # --- HELPER: Run YOLOv8 (Cleaned Name) ---
    def run_yolov8(self, img):
        if not self.model_v8: return []
        
        # Ultralytics handles resizing/preprocessing automatically
        results = self.model_v8(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        
        boxes_list = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                boxes_list.append([coords[0], coords[1], coords[2], coords[3], conf])
        
        return boxes_list

    # --- HELPER: NMS ---
    def nms(self, boxes, scores, thresh):
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    # --- HELPER: Smart Context-Aware OCR ---
    def ensemble_ocr(self, plate_crop):
        candidates = []
        VALID_PROVINCES = ["WP", "EP", "NP", "SP", "SG", "NC", "CP", "UP", "NW"]

        def fix_province_code(text):
            text = text.upper().replace(" ", "")
            for prov in VALID_PROVINCES:
                if prov in text: return text
            fixes = {"5P": "SP", "8P": "SP", "S9": "SP", "W9": "WP", "VP": "WP", "VV": "WP", "NWC": "NW"}
            for wrong, right in fixes.items():
                if wrong in text: return text.replace(wrong, right)
            return text

        def get_smart_text(image_input):
            w, h = image_input.size
            image_input = image_input.resize((w * 2, h * 2), Image.LANCZOS)
            img_arr = np.array(image_input)

            results = self.reader.readtext(img_arr, detail=1)
            if not results: return ""
            results = [r for r in results if r[2] > 0.30]
            if not results: return ""

            y_centers = [(r[0][0][1] + r[0][2][1]) / 2 for r in results]
            height_span = max(y_centers) - min(y_centers)
            is_stacked = height_span > (img_arr.shape[0] * 0.35)

            if is_stacked:
                results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                mid_y = img_arr.shape[0] / 2
                top_part = []
                bottom_part = []
                
                for r in results:
                    y_c = (r[0][0][1] + r[0][2][1]) / 2
                    val = r[1].upper().replace(".", "").replace("-", "")
                    if y_c < mid_y:
                        top_part.append(val)
                    else:
                        val = val.replace("O", "0").replace("I", "1").replace("B", "8")
                        bottom_part.append(val)
                return fix_province_code("".join(top_part)) + " " + "".join(bottom_part)
            else:
                results.sort(key=lambda r: r[0][0][0])
                raw = "".join([r[1] for r in results]).upper()
                clean = raw.replace("-", "").replace(".", "").replace(" ", "")
                return fix_province_code(clean)

        candidates.append(get_smart_text(plate_crop))
        gray = ImageOps.grayscale(plate_crop)
        enhancer = ImageEnhance.Contrast(gray)
        high_contrast = enhancer.enhance(2.0)
        candidates.append(get_smart_text(high_contrast))

        cleaned = [c for c in candidates if len(c) > 3]
        if not cleaned: return "Unknown"
        return Counter(cleaned).most_common(1)[0][0]

    def draw_detections(self, image_bytes, detections):
        img = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(img)
        for det in detections:
            b = det["bbox"]
            text = det.get("text", "")
            draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="magenta", width=4)
            if text:
                draw.rectangle([b["x1"], b["y1"]-25, b["x1"]+150, b["y1"]], fill="magenta")
                draw.text((b["x1"]+5, b["y1"]-20), f"{text} ({det['confidence']})", fill="white")
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

# Initialize global instance
detector = AdvancedYOLOService()