import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw
import io
from app.config import MODEL_PATH, CONFIDENCE_THRESHOLD

class YOLOService:
    def __init__(self):
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = (640, 640)

    def load_model(self):
        """Loads the ONNX model into an inference session."""
        try:
            print(f"🚀 Loading ONNX model from {MODEL_PATH}...")
            # Load the model with CPU provider
            self.session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            
            # Get input and output details
            model_inputs = self.session.get_inputs()
            self.input_name = model_inputs[0].name
            
            model_outputs = self.session.get_outputs()
            self.output_names = [output.name for output in model_outputs]
            
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.session = None

    def preprocess(self, image_bytes: bytes):
        """
        Resizes and normalizes the image for YOLO.
        Returns: (preprocessed_image, original_size)
        """
        # 1. Load Image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        original_width, original_height = img.size
        
        # 2. Resize (Simple resize to 640x640)
        img_resized = img.resize(self.input_shape)
        
        # 3. Convert to Numpy and Normalize
        img_data = np.array(img_resized).astype(np.float32)
        img_data /= 255.0  # Normalize to 0-1
        
        # 4. Transpose to (Batch, Channel, Height, Width) -> (1, 3, 640, 640)
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data, (original_width, original_height)

    def predict(self, image_bytes: bytes) -> list:
        if not self.session:
            raise RuntimeError("Model is not loaded")

        # 1. Preprocess
        input_tensor, (orig_w, orig_h) = self.preprocess(image_bytes)

        # 2. Inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # Output shape is typically [1, 5, 8400] for 1 class (x, y, w, h, conf)
        predictions = np.squeeze(outputs[0]).T

        # 3. Post-Processing (Filtering)
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > CONFIDENCE_THRESHOLD, :]
        scores = scores[scores > CONFIDENCE_THRESHOLD]

        if len(predictions) == 0:
            return []

        # Get the class with highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # 4. Non-Max Suppression (NMS) & Coordinate Scaling
        boxes = predictions[:, :4]
        indices = self.nms(boxes, scores, iou_threshold=0.45)

        processed_detections = []
        x_scale = orig_w / self.input_shape[0]
        y_scale = orig_h / self.input_shape[1]

        for i in indices:
            # Convert cx, cy, w, h to x1, y1, x2, y2
            cx, cy, w, h = boxes[i]
            
            # Scale back to original image size
            cx *= x_scale
            cy *= y_scale
            w *= x_scale
            h *= y_scale

            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)

            processed_detections.append({
                "class_name": "license_plate",
                "confidence": round(float(scores[i]), 2),
                "bbox": {
                    "x1": max(0, x1),
                    "y1": max(0, y1),
                    "x2": min(orig_w, x2),
                    "y2": min(orig_h, y2)
                }
            })
        
        return processed_detections

    def draw_detections(self, image_bytes: bytes, detections: list) -> bytes:
        """
        Draws bounding boxes on the image and returns JPEG bytes.
        """
        # Load image
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        draw = ImageDraw.Draw(img)
        
        for det in detections:
            box = det["bbox"]
            x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
            
            # Draw Red Box (width=5 for thickness)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=5)
            
            # Draw Text (Score)
            text = f"{det['confidence']}"
            # Draw text slightly above the box
            draw.text((x1, y1 - 10), text, fill="red")
            
        # Save to bytes
        output = io.BytesIO()
        img.save(output, format="JPEG")
        return output.getvalue()

    def nms(self, boxes, scores, iou_threshold):
        """
        Perform Non-Maximum Suppression (NMS) on the boxes.
        """
        # Convert to x1, y1, x2, y2 for NMS calculation
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2

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
            
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

# Create a global instance
detector = YOLOService()