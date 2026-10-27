from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routes.predict import router as predict_router
from routes.heatmap import router as heatmap_router
from routes.zones import router as zones_router
from routes.status import router as status_router


app = FastAPI(title="Forest Fire Prediction and Early Warning System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(heatmap_router)
app.include_router(zones_router)
app.include_router(status_router)


@app.get("/")
def root():
    return {"message": "Fire Prediction API is running", "docs": "/docs", "status": "/status"}
