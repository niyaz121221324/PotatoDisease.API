from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from app.model.load_model import load_model
from app.utils.preprocess import preprocess_image
from PIL import Image, UnidentifiedImageError
import json
import numpy as np
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))

# Пути к метаданным
METADATA_PATH = os.path.join(current_dir, 'metadata/metadata.json')

# Загрузка модели и метаданных
model = load_model()

def serialize_to_dict(json_data):
    class_indices = json_data["class_indices"]
    serialized_dict = {value: key.split("-", 1)[1] for key, value in class_indices.items()}
    return serialized_dict

with open(METADATA_PATH, "r") as f:
    class_indices = json.load(f)

class_names = serialize_to_dict(class_indices)

@app.get("/")
def root():
    return {"message": "Welcome to the Image Prediction API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        preprocessed_image = preprocess_image(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    finally:
        file.file.close()

    try:
        predictions = model.predict(preprocessed_image)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {e}")

    return JSONResponse(
        content={
            "className": class_names[predicted_class],
            "confidence": confidence,
        }
    )