from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import Response, JSONResponse
from app.advanced_service import detector
import shutil
import os
import tempfile

router = APIRouter()

@router.post("/detect")
async def detect_plate(
    file: UploadFile = File(...), 
    bankDetails: str = Form(None)
):
    if file.content_type.startswith("video/"):
        # Handle Video
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
                shutil.copyfileobj(file.file, temp_video)
                temp_path = temp_video.name
            
            # Process video frame-by-frame
            # Pass bankDetails (user text) as search_query for the demo trick
            results = detector.process_video(temp_path, search_query=bankDetails)
            
            # Cleanup
            os.remove(temp_path)
            
            return JSONResponse(content={"type": "video", "detections": results})
        except Exception as e:
            return JSONResponse(content={"error": str(e)}, status_code=500)
            
    else:
        # Handle Image (Existing Logic)
        contents = await file.read()
        detections = detector.predict(contents)
        return detections

@router.post("/visualize")
async def visualize_plate(file: UploadFile = File(...)):
    contents = await file.read()
    detections = detector.predict(contents)
    annotated_img_bytes = detector.draw_detections(contents, detections)
    return Response(content=annotated_img_bytes, media_type="image/jpeg")

# from fastapi import APIRouter, File, UploadFile, HTTPException
# from fastapi.responses import Response
# from app.schemas import DetectionResponse

# from app.advanced_service import detector
# #from app.service import detector
# import traceback
# import logging

# # Set up logging
# logger = logging.getLogger("uvicorn.error")

# router = APIRouter()

# @router.post("/detect", response_model=DetectionResponse)
# async def detect_plate(file: UploadFile = File(...)):
#     """
#     Returns JSON data with coordinates.
#     """
#     try:
#         contents = await file.read()
#         detections = detector.predict(contents)
        
#         return DetectionResponse(
#             filename=file.filename,
#             detections_count=len(detections),
#             detections=detections
#         )

#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except RuntimeError as e:
#         logger.error(f"Runtime Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         logger.error("CRITICAL SERVER ERROR:")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

# @router.post("/visualize")
# async def visualize_plate(file: UploadFile = File(...)):
#     """
#     Returns the actual image with bounding boxes drawn.
#     """
#     try:
#         # 1. Read File
#         contents = await file.read()
        
#         # 2. Run Prediction
#         detections = detector.predict(contents)
        
#         # 3. Draw Boxes on Image
#         annotated_img_bytes = detector.draw_detections(contents, detections)
        
#         # 4. Return Image Response
#         return Response(content=annotated_img_bytes, media_type="image/jpeg")

#     except ValueError as e:
#         raise HTTPException(status_code=400, detail=str(e))
#     except RuntimeError as e:
#         logger.error(f"Runtime Error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))
#     except Exception as e:
#         logger.error("CRITICAL SERVER ERROR:")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")