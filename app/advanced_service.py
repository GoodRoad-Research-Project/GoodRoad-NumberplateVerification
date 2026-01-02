import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import io
import easyocr
from collections import Counter
from ultralytics import YOLO
from app.config import MODEL_PATH_V9, MODEL_PATH_V8, MODEL_PATH_SRGAN, CONFIDENCE_THRESHOLD

class AdvancedYOLOService:
    def __init__(self):
        self.session_v9 = None
        self.model_v8 = None
        self.session_srgan = None
        self.input_name_srgan = None
        
        print("🚀 Initializing Super-Resolution Hybrid Engine...")
        
        # 1. Initialize EasyOCR
        self.reader = easyocr.Reader(['en'], gpu=False) 
        
        # 2. Load All Models
        self.load_models()

    def load_models(self):
        """Loads SRGAN, YOLOv9, and YOLOv8"""
        # --- Load SRGAN (Enhancer) ---
        try:
            print(f"✨ Loading SRGAN from {MODEL_PATH_SRGAN}...")
            self.session_srgan = ort.InferenceSession(MODEL_PATH_SRGAN, providers=["CPUExecutionProvider"])
            self.input_name_srgan = self.session_srgan.get_inputs()[0].name
            print("✅ SRGAN Loaded")
        except Exception as e:
            print(f"❌ Error loading SRGAN: {e}")

        # --- Load YOLOv9 ---
        try:
            print(f"🚀 Loading YOLOv9 from {MODEL_PATH_V9}...")
            self.session_v9 = ort.InferenceSession(MODEL_PATH_V9, providers=["CPUExecutionProvider"])
            self.input_name_v9 = self.session_v9.get_inputs()[0].name
            self.output_names_v9 = [o.name for o in self.session_v9.get_outputs()]
            print("✅ YOLOv9 Loaded")
        except Exception as e:
            print(f"❌ Error loading YOLOv9: {e}")

        # --- Load YOLOv8 ---
        try:
            print(f"🚀 Loading YOLOv8 from {MODEL_PATH_V8}...")
            self.model_v8 = YOLO(MODEL_PATH_V8, task='detect')
            print("✅ YOLOv8 Loaded")
        except Exception as e:
            print(f"❌ Error loading YOLOv8: {e}")

    def enhance_image(self, img):
        """
        Runs the image through SRGAN to upscale and de-blur.
        """
        if not self.session_srgan:
            return img  # Skip if model failed to load

        # 1. Preprocess: Resize if too huge (to prevent crash), normalize to 0-1
        # SRGAN is heavy; if image is > 1000px, we limit input size for speed
        max_dim = 1024
        if max(img.size) > max_dim:
             ratio = max_dim / max(img.size)
             new_size = (int(img.width * ratio), int(img.height * ratio))
             img = img.resize(new_size, Image.LANCZOS)

        img_np = np.array(img).astype(np.float32) / 255.0
        img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
        img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension -> 1,C,H,W

        # 2. Inference
        try:
            output = self.session_srgan.run(None, {self.input_name_srgan: img_np})[0]
        except Exception as e:
            print(f"⚠️ SRGAN Failed: {e}")
            return img

        # 3. Postprocess: Clip values, convert back to uint8 image
        output = output.squeeze(0).transpose(1, 2, 0)  # CHW -> HWC
        output = np.clip(output, 0, 1) * 255.0
        output = output.astype(np.uint8)
        
        return Image.fromarray(output)

    def predict(self, image_bytes: bytes):
        """
        Flow: Original -> SRGAN -> Enhanced -> Ensemble Detect -> OCR -> Scale Boxes Back
        """
        # 1. Load Original Image
        original_img = Image.open(io.BytesIO(image_bytes))
        if original_img.mode != "RGB":
            original_img = original_img.convert("RGB")
            
        # 2. ✨ ENHANCE IMAGE (SRGAN) ✨
        # This creates a larger, sharper version of the image
        enhanced_img = self.enhance_image(original_img)
        
        # 3. Run Detectors on Enhanced Image
        boxes_v9 = self.run_yolov9_onnx(enhanced_img)
        boxes_v8 = self.run_yolov8(enhanced_img)
        
        # 4. Ensemble Fusion (NMS)
        all_boxes = boxes_v9 + boxes_v8
        if not all_boxes: return []
        
        all_boxes_np = np.array(all_boxes)
        keep_indices = self.nms(all_boxes_np[:, :4], all_boxes_np[:, 4], 0.45)
        
        final_detections = []
        
        # Calculate scaling factor to map Enhanced coordinates back to Original coordinates
        scale_x = original_img.width / enhanced_img.width
        scale_y = original_img.height / enhanced_img.height

        for i in keep_indices:
            # Box on the ENHANCED image
            bbox_enhanced = all_boxes_np[i, :4].astype(int)
            conf = float(all_boxes_np[i, 4])
            
            # 5. Context-Aware OCR (Using the High-Res Enhanced Crop)
            plate_crop = enhanced_img.crop((bbox_enhanced[0], bbox_enhanced[1], bbox_enhanced[2], bbox_enhanced[3]))
            plate_text = self.ensemble_ocr(plate_crop)
            
            # 6. Map coordinates back to ORIGINAL image size for the UI
            bbox_original = {
                "x1": int(bbox_enhanced[0] * scale_x),
                "y1": int(bbox_enhanced[1] * scale_y),
                "x2": int(bbox_enhanced[2] * scale_x),
                "y2": int(bbox_enhanced[3] * scale_y)
            }
            
            final_detections.append({
                "class_name": "License Plate",
                "confidence": round(conf, 2),
                "bbox": bbox_original,
                "text": plate_text
            })
            
        return final_detections

    # --- HELPER: Run YOLOv9 ---
    def run_yolov9_onnx(self, img):
        if not self.session_v9: return []
        img_resized = img.resize((640, 640))
        input_data = np.array(img_resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        input_data = input_data[None, ...]
        outputs = self.session_v9.run(self.output_names_v9, {self.input_name_v9: input_data})
        predictions = np.squeeze(outputs[0]).T
        boxes_list = []
        img_w, img_h = img.size
        for pred in predictions:
            if pred[4] > CONFIDENCE_THRESHOLD:
                cx, cy, w, h = pred[0], pred[1], pred[2], pred[3]
                x1, y1 = (cx - w/2) * (img_w / 640), (cy - h/2) * (img_h / 640)
                x2, y2 = (cx + w/2) * (img_w / 640), (cy + h/2) * (img_h / 640)
                boxes_list.append([x1, y1, x2, y2, pred[4]])
        return boxes_list

    # --- HELPER: Run YOLOv8 ---
    def run_yolov8(self, img):
        if not self.model_v8: return []
        results = self.model_v8(img, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes_list = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].cpu().numpy()
                boxes_list.append([coords[0], coords[1], coords[2], coords[3], float(box.conf[0])])
        return boxes_list

    # --- HELPER: NMS ---
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

    # --- HELPER: Smart OCR ---
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
            is_stacked = (max(y_centers) - min(y_centers)) > (img_arr.shape[0] * 0.35)
            if is_stacked:
                results.sort(key=lambda r: (r[0][0][1], r[0][0][0]))
                mid_y = img_arr.shape[0] / 2
                top, bottom = [], []
                for r in results:
                    y_c = (r[0][0][1] + r[0][2][1]) / 2
                    val = r[1].upper().replace(".", "").replace("-", "")
                    if y_c < mid_y: top.append(val)
                    else: bottom.append(val.replace("O", "0").replace("I", "1").replace("B", "8"))
                return fix_province_code("".join(top)) + " " + "".join(bottom)
            else:
                results.sort(key=lambda r: r[0][0][0])
                return fix_province_code("".join([r[1] for r in results]).upper().replace("-", "").replace(".", "").replace(" ", ""))

        candidates.append(get_smart_text(plate_crop))
        gray = ImageOps.grayscale(plate_crop)
        candidates.append(get_smart_text(ImageEnhance.Contrast(gray).enhance(2.0)))
        cleaned = [c for c in candidates if len(c) > 3]
        if not cleaned: return "Unknown"
        return Counter(cleaned).most_common(1)[0][0]

    def draw_detections(self, image_bytes, detections):
        img = Image.open(io.BytesIO(image_bytes))
        draw = ImageDraw.Draw(img)
        for det in detections:
            b = det["bbox"]
            text = det.get("text", "")
            draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="#00ff00", width=4)
            if text:
                draw.rectangle([b["x1"], b["y1"]-25, b["x1"]+150, b["y1"]], fill="#00ff00")
                draw.text((b["x1"]+5, b["y1"]-20), f"{text} ({det['confidence']})", fill="black")
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

detector = AdvancedYOLOService()
