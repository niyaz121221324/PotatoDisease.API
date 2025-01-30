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
LEAF_METADATA_PATH = os.path.join(current_dir, 'metadata/leaves_metadata.json')
TUBER_METADATA_PATH = os.path.join(current_dir, 'metadata/tuber_metadata.json')

# Загрузка модели и метаданных
binary_model = load_model('general_classifier.h5')
tuber_model = load_model('tuber.h5')
leaf_model = load_model('leaves.h5')

def serialize_to_dict(json_data):
    class_indices = json_data["class_indices"]
    serialized_dict = {value: key for key, value in class_indices.items()}
    return serialized_dict

def load_class_indicies(metadata_path: str):
    with open(metadata_path, "r") as f:
        return json.load(f)

leaf_class_names = serialize_to_dict(load_class_indicies(LEAF_METADATA_PATH))
tuber_class_names = serialize_to_dict(load_class_indicies(TUBER_METADATA_PATH))

@app.get("/")
def root():
    return {"message": "Welcome to the Image Prediction API!"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Открываем и предобрабатываем изображение
        image = Image.open(file.file)
        preprocessed_image = preprocess_image(image)
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {e}")
    finally:
        file.file.close()

    try:
        # Определяем, клубень это или лист
        category_pred = binary_model.predict(preprocessed_image)[0][0]
        model, class_names = (leaf_model, leaf_class_names) if category_pred < 0.5 else (tuber_model, tuber_class_names)

        # Выполняем предсказание
        predictions = model.predict(preprocessed_image)
        predicted_class = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_class])

        return JSONResponse(
            content={ 
                "predictedClass": class_names[predicted_class], 
                "confidence": confidence
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making predictions: {e}")