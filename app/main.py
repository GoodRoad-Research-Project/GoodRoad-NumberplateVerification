from fastapi import FastAPI, Response
from contextlib import asynccontextmanager
from app.routes import router
#from app.service import detector
from app.advanced_service import detector

#for easyocr
from fastapi.middleware.cors import CORSMiddleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    detector.load_model()
    yield

app = FastAPI(
    title="Structured License Plate API",
    version="2.0",
    lifespan=lifespan
)

# --- CORS SETUP ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- ADD THIS PART ---
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


app.include_router(router)

@app.get("/")
def root():
    return {"status": "online", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)