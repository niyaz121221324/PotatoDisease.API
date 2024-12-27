import tensorflow as tf
import os

def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Формируем относительный путь к модели
    model_path = os.path.join(current_dir, 'potato_0heal_1light_2late.h5')

    # Загружаем модель
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        raise FileNotFoundError(f"The model file at {model_path} does not exist.")