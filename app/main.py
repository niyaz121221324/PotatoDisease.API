from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.model.load_model import load_model
from app.utils.preprocess import preprocess_image
from PIL import Image
import json
import numpy as np
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к метаданным
METADATA_PATH = os.path.join(current_dir, 'metadata/metadata.json')

# Загрузка модели и метаданных
model = load_model()

with open(METADATA_PATH, "r") as f:
    class_indices = json.load(f)

class_names = class_indices['class_indices']

@app.get("/")
def root():
    return {"message": "Welcome to the Image Prediction API!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        preprocessed_image = preprocess_image(image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    return JSONResponse(
        content={
            "class": class_names[predicted_class],
            "confidence": float(confidence),
        }
    )