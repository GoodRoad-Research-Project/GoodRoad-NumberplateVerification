from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import Response
from app.schemas import DetectionResponse

from app.advanced_service import detector
#from app.service import detector
import traceback
import logging

# Set up logging
logger = logging.getLogger("uvicorn.error")

router = APIRouter()

@router.post("/detect", response_model=DetectionResponse)
async def detect_plate(file: UploadFile = File(...)):
    """
    Returns JSON data with coordinates.
    """
    try:
        contents = await file.read()
        detections = detector.predict(contents)
        
        return DetectionResponse(
            filename=file.filename,
            detections_count=len(detections),
            detections=detections
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("CRITICAL SERVER ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

@router.post("/visualize")
async def visualize_plate(file: UploadFile = File(...)):
    """
    Returns the actual image with bounding boxes drawn.
    """
    try:
        # 1. Read File
        contents = await file.read()
        
        # 2. Run Prediction
        detections = detector.predict(contents)
        
        # 3. Draw Boxes on Image
        annotated_img_bytes = detector.draw_detections(contents, detections)
        
        # 4. Return Image Response
        return Response(content=annotated_img_bytes, media_type="image/jpeg")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Runtime Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error("CRITICAL SERVER ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")