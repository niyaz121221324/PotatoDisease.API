# PotatoDesease.API

Это проект API на FastAPI, который оборачивает модель машинного обучения для анализа качества картофеля по его листьям и клубням. Проект использует TensorFlow и Docker для контейнеризации.

## Описание

API принимает изображения листьев и клубней картофеля и классифицирует их по категориям:
- Бактериальная мягкая гниль картофеля (`Bacterial_soft_rot_of_potato`)
- Здоровый по листьям (`Healthy`)
- Ранний фитофтороз по листьям (`Early_Blight`)
- Поздний фитофтороз по листьям (`Late_Blight`)
- Антракноз картофеля (`Potato_anthracnose`)
- Бактериальное увядание картофеля (`Potato_bacterial_wilt`)
- Чёрная парша картофеля (`Potato_black_scurf`)
- Чёрная ножка картофеля (`Potato_black_shank_disease`)
- Обыкновенная язва картофеля (`Potato_common_scab`)
- Сухая гниль картофеля (`Potato_dry_rot`)
- Ранний фитофтороз по клубням (`Potato_early_blight`)
- Поздний фитофтороз по клубням (`Potato_late_blight`)
- Вертициллезное увядание картофеля (`Potato_verticillium_wilt`)
- Бородавчатая болезнь картофеля (`Potato_wart_disease`)
- Увядание картофеля (`Potato_wilt`)
- Порошковая язва картофеля (`Powdery_scab_of_potato`)
- кольцевая бактериальная гниль картофеля (`potato_bacterial_ring_rot`)

## Метаданные

Файл `metadata.json` содержит информацию о классах:

```json
{
    "class_indices": 
    {
        "Bacterial_soft_rot_of_potato": 0, 
        "Potato___Early_blight": 1, 
        "Potato___Late_blight": 2, 
        "Potato___healthy": 3, 
        "Potato_anthracnose": 4, 
        "Potato_bacterial_wilt": 5, 
        "Potato_black_scurf": 6, 
        "Potato_black_shank_disease": 7, 
        "Potato_common_scab": 8, 
        "Potato_dry_rot": 9, 
        "Potato_early_blight": 10, 
        "Potato_late_blight": 11, 
        "Potato_verticillium_wilt": 12, 
        "Potato_wart_disease": 13, 
        "Potato_wilt": 14, 
        "Powdery_scab_of_potato": 15, 
        "potato_bacterial_ring_rot": 16
    }
}
```

# Установка и запуск

### Клонируйте репозиторий:
```sh
git clone https://github.com/niyaz121221324/PotatoDisease.API.git
cd PotatoDisease.API
```

### Запуск приложения
```sh
docker-compose up --build
```

### Пример Использования
### Загрузка изображения для анализа
Отправьте POST запрос на /predict с изображением листа картофеля:
```sh
curl -X POST "http://localhost:8080/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

### Ответ сервера
Ответ будет содержать предсказанный класс и вероятность:
```json
{
    "predictedClass": "1-Early_Blight",
    "confidence": 0.95
}
```
