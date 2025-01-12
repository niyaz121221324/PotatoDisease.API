# PotatoDesease.API

Это проект API на FastAPI, который оборачивает модель машинного обучения для анализа качества картофеля по его листьям. Проект использует TensorFlow и Docker для контейнеризации.

## Описание

API принимает изображения листьев картофеля и классифицирует их по трём категориям:
- Здоровый (`0-Healthy`)
- Ранний фитофтороз (`1-Early_Blight`)
- Поздний фитофтороз (`2-Late_Blight`)

## Метаданные

Файл `metadata.json` содержит информацию о классах:

```json
{
    "class_indices": 
    {
        "0-Healthy": 0, 
        "1-Early_Blight": 1, 
        "2-Late_Blight": 2
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
