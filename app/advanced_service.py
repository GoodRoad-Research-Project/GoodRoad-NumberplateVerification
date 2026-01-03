import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import easyocr
from collections import Counter
from ultralytics import YOLO
from app.config import MODEL_PATH_V9, MODEL_PATH_V8, MODEL_PATH_SRGAN, CONFIDENCE_THRESHOLD
import cv2  # Needs: pip install opencv-python
import re

class AdvancedYOLOService:
    def __init__(self):
        self.session_v9 = None
        self.model_v8 = None
        self.session_srgan = None
        self.input_name_srgan = None
        
        print("🚀 Initializing Final Hybrid Engine (Otsu + Regex)...")
        # Optimized for English text detection
        self.reader = easyocr.Reader(['en'], gpu=False) 
        self.load_models()

    def load_models(self):
        # --- SRGAN ---
        try:
            print(f"✨ Loading SRGAN from {MODEL_PATH_SRGAN}...")
            self.session_srgan = ort.InferenceSession(MODEL_PATH_SRGAN, providers=["CPUExecutionProvider"])
            self.input_name_srgan = self.session_srgan.get_inputs()[0].name
            print("✅ SRGAN Loaded")
        except Exception as e:
            print(f"❌ SRGAN Error: {e}")
            self.session_srgan = None

        # --- YOLOv9 ---
        try:
            print(f"🚀 Loading YOLOv9 from {MODEL_PATH_V9}...")
            self.session_v9 = ort.InferenceSession(MODEL_PATH_V9, providers=["CPUExecutionProvider"])
            self.input_name_v9 = self.session_v9.get_inputs()[0].name
            self.output_names_v9 = [o.name for o in self.session_v9.get_outputs()]
            print("✅ YOLOv9 Loaded")
        except Exception as e:
            print(f"❌ YOLOv9 Error: {e}")

        # --- YOLOv8 ---
        try:
            print(f"🚀 Loading YOLOv8 from {MODEL_PATH_V8}...")
            self.model_v8 = YOLO(MODEL_PATH_V8, task='detect')
            print("✅ YOLOv8 Loaded")
        except Exception as e:
            print(f"❌ YOLOv8 Error: {e}")

    def enhance_plate_crop(self, plate_img):
        """
        Smart Enhance:
        1. If plate is SMALL (<64px), use SRGAN to upscale.
        2. If plate is BIG (>64px), SKIP SRGAN (prevent shrinking/quality loss).
        """
        if not self.session_srgan: return plate_img

        # SMART CHECK: If image is already good, don't ruin it by shrinking!
        if plate_img.width > 90: # 90 is a safe threshold
            return plate_img

        try:
            # 1. Pad to Square (Preserve Aspect Ratio)
            old_size = plate_img.size
            new_size = (max(old_size), max(old_size))
            new_im = Image.new("RGB", new_size, (0, 0, 0))
            new_im.paste(plate_img, ((new_size[0]-old_size[0])//2, (new_size[1]-old_size[1])//2))

            # 2. Resize to 64x64 (Strict Model Requirement)
            input_img = new_im.resize((64, 64), Image.BICUBIC)
            
            # 3. Inference
            img_np = np.array(input_img).astype(np.float32) / 255.0
            img_np = img_np.transpose(2, 0, 1)
            img_np = np.expand_dims(img_np, axis=0)

            output = self.session_srgan.run(None, {self.input_name_srgan: img_np})[0]

            # 4. Post-Process
            output = output.squeeze(0).transpose(1, 2, 0)
            if output.min() < 0: output = (output + 1) / 2.0
            output = np.clip(output, 0, 1) * 255.0
            output = output.astype(np.uint8)
            
            return Image.fromarray(output)
            
        except Exception as e:
            print(f"⚠️ Enhancement Failed: {e}")
            return plate_img

    def apply_ocr_filters(self, pil_img):
        """
        Applies Otsu's Thresholding to make text 'pop' against the background.
        Crucial for Sri Lankan plates with shadows/glare.
        """
        # Convert PIL to OpenCV format
        img = np.array(pil_img) 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 1. Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Gaussian Blur (Remove noise dots)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # 3. Otsu's Thresholding (Auto-Binarization)
        # This converts gray blobs into crisp black/white shapes
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary # Returns numpy array for EasyOCR

    def correct_sri_lankan_text(self, text):
        """
        Uses Regex to force 'WP CA 1234' format.
        Fixes 0 vs O, 8 vs B, etc.
        """
        # Remove special chars (keep only A-Z and 0-9)
        clean = re.sub(r'[^A-Z0-9]', '', text.upper())
        
        # Logic: Sri Lankan plates usually have Letters first, Numbers last
        # If length is roughly 6-7 chars (e.g. WPCA1234)
        if len(clean) >= 6:
            prefix = clean[:2]   # First 2 chars (Province? WP, SP)
            middle = clean[2:-4] # Middle Letters (CA, CAB)
            suffix = clean[-4:]  # Last 4 Numbers (1234)

            # Fix Prefix (Letters) - e.g. 0 -> O, 8 -> B
            prefix = prefix.replace('0', 'O').replace('1', 'I').replace('4', 'A').replace('8', 'B')
            
            # Fix Middle (Letters)
            middle = middle.replace('0', 'O').replace('1', 'I').replace('8', 'B')
            
            # Fix Suffix (Numbers) - e.g. O -> 0, I -> 1, B -> 8
            suffix = suffix.replace('O', '0').replace('I', '1').replace('B', '8').replace('S', '5').replace('Z', '2')

            return f"{prefix} {middle} {suffix}"
        
        return text

    def predict(self, image_bytes: bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # 1. Detect
        boxes_v9 = self.run_yolov9_onnx(img)
        boxes_v8 = self.run_yolov8(img)
        all_boxes = boxes_v9 + boxes_v8
        if not all_boxes: return []
        
        # 2. Aggressive NMS (Merge 'Ghost' boxes)
        keep_indices = self.nms(np.array(all_boxes)[:, :4], np.array(all_boxes)[:, 4], 0.30)
        
        final_detections = []
        all_boxes_np = np.array(all_boxes)
        
        # Limit to top 3 detections to avoid garbage
        sorted_indices = sorted(keep_indices, key=lambda i: all_boxes_np[i, 4], reverse=True)[:3]

        for idx in sorted_indices:
            bbox = all_boxes_np[idx, :4].astype(int)
            conf = float(all_boxes_np[idx, 4])
            
            # Crop
            margin = 5
            x1, y1 = max(0, bbox[0]-margin), max(0, bbox[1]-margin)
            x2, y2 = min(img.width, bbox[2]+margin), min(img.height, bbox[3]+margin)
            plate_crop = img.crop((x1, y1, x2, y2))
            
            # --- PIPELINE ---
            plate_text = "Unknown"
            
            # 1. Enhance (SRGAN)
            enhanced_pil = self.enhance_plate_crop(plate_crop)
            
            # 2. Filter (Otsu Thresholding)
            processed_img = self.apply_ocr_filters(enhanced_pil)
            
            # 3. Read Text
            raw_text = self.read_text_easyocr(processed_img)
            
            # 4. Correct Text (Regex)
            if len(raw_text) > 3:
                plate_text = self.correct_sri_lankan_text(raw_text)

            # Fallback: If "Unknown", try original crop without filters
            if plate_text == "Unknown":
                raw_text_orig = self.read_text_easyocr(np.array(plate_crop))
                if len(raw_text_orig) > 3:
                     plate_text = self.correct_sri_lankan_text(raw_text_orig)

            final_detections.append({
                "class_name": "License Plate",
                "confidence": round(conf, 2),
                "bbox": {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])},
                "text": plate_text
            })
            
        return final_detections

    def read_text_easyocr(self, img_input):
        result = self.reader.readtext(img_input, detail=0)
        return "".join(result)

    # --- Helpers ---
    def run_yolov9_onnx(self, img):
        if not self.session_v9: return []
        img_resized = img.resize((640, 640))
        input_data = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        input_data = input_data[None, ...]
        outputs = self.session_v9.run(self.output_names_v9, {self.input_name_v9: input_data})
        predictions = np.squeeze(outputs[0]).T
        boxes = []
        w_scale, h_scale = img.size[0]/640, img.size[1]/640
        for p in predictions:
            if p[4] > CONFIDENCE_THRESHOLD:
                cx, cy, w, h = p[0], p[1], p[2], p[3]
                boxes.append([(cx-w/2)*w_scale, (cy-h/2)*h_scale, (cx+w/2)*w_scale, (cy+h/2)*h_scale, p[4]])
        return boxes

    def run_yolov8(self, img):
        if not self.model_v8: return []
        results = self.model_v8(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = []
        for r in results:
            for b in r.boxes:
                c = b.xyxy[0].cpu().numpy()
                boxes.append([c[0], c[1], c[2], c[3], float(b.conf[0])])
        return boxes

    def nms(self, boxes, scores, thresh):
        if len(boxes) == 0: return []
        x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
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
            w, h = np.maximum(0.0, xx2 - xx1), np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def draw_detections(self, image_bytes, detections):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)
        for det in detections:
            b = det["bbox"]
            text = det.get("text", "")
            draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="cyan", width=4)
            if text:
                draw.rectangle([b["x1"], b["y1"]-25, b["x1"]+150, b["y1"]], fill="cyan")
                draw.text((b["x1"]+5, b["y1"]-20), f"{text} ({det['confidence']})", fill="black")
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

detector = AdvancedYOLOService()