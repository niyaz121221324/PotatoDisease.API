# PotatoDesease.API

Это проект API на FastAPI, который оборачивает модель машинного обучения для анализа качества картофеля по его листьям и клубням. Проект использует TensorFlow и Docker для контейнеризации.

## Описание

API принимает изображения листьев и клубней, сначала определяя их категорию с помощью модели general_classifier. В зависимости от результата, изображения классифицируются по соответствующей модели: для листьев — одна, для клубней — другая.

Анализ клубней:
- Бактериальная мягкая гниль картофеля (`Bacterial_soft_rot_of_potato`)
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

## Метаданные клубней

Файл `tuber_metadata.json` содержит информацию о классах по клубням:

```json
{
  "class_indices": 
  {
    "Bacterial_soft_rot_of_potato": 0,
    "Potato_anthracnose": 1,
    "Potato_bacterial_wilt": 2,
    "Potato_black_scurf": 3,
    "Potato_black_shank_disease": 4,
    "Potato_common_scab": 5,
    "Potato_dry_rot": 6,
    "Potato_early_blight": 7,
    "Potato_late_blight": 8,
    "Potato_verticillium_wilt": 9,
    "Potato_wart_disease": 10,
    "Potato_wilt": 11,
    "Powdery_scab_of_potato": 12,
    "potato_bacterial_ring_rot": 13
  }
}
  
```
Анализ 
- Здоровый по листьям (`Healthy`)
- Ранний фитофтороз по листьям (`Early_Blight`)
- Поздний фитофтороз по листьям (`Late_Blight`)

## Метаданные

Файл `leaves_metadata.json` содержит информацию о классах листьев:

```json
{
    "class_indices": 
    {
        "Potato___Early_blight": 0, 
        "Potato___Late_blight": 1, 
        "Potato___healthy": 2
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
Отправьте POST запрос на /predict с изображением листа или клубня картофеля:
```sh
curl -X POST "http://localhost:8080/predict" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

### Ответ сервера
Ответ будет содержать предсказанный класс и вероятность:
```json
{
    "predictedClass": "Potato___Early_blight",
    "confidence": 0.95
}
```
