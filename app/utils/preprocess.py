from PIL import Image
import numpy as np

# Нормализируем изображение
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size, Image.LANCZOS)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)