# from pydantic import BaseModel
# from typing import List

# class BoundingBox(BaseModel):
#     x1: int
#     y1: int
#     x2: int
#     y2: int

# class Detection(BaseModel):
#     class_name: str
#     confidence: float
#     bbox: BoundingBox

# class DetectionResponse(BaseModel):
#     filename: str
#     detections_count: int
#     detections: List[Detection]

from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class Detection(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox
    text: Optional[str] = None

class DetectionResponse(BaseModel):
    filename: str
    detections_count: int
    detections: List[Detection]
    image_base64: Optional[str] = None 