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