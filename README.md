# PotatoDisease.API

**PotatoDisease.API** — это API на **FastAPI**, которое оборачивает модель машинного обучения для анализа качества картофеля по его листьям и клубням. 
Проект использует **TensorFlow** и **Docker** для контейнеризации.

---
## Описание

API принимает изображения **листьев** и **клубней** картофеля:
1. **General Classifier** определяет категорию изображения (лист или клубень).
2. В зависимости от категории используется соответствующая модель:
   - **Модель для листьев**
   - **Модель для клубней**

### Заболевания клубней
API распознает следующие заболевания картофеля:

- **Бактериальная мягкая гниль** (*Bacterial_soft_rot_of_potato*)
- **Антракноз** (*Potato_anthracnose*)
- **Бактериальное увядание** (*Potato_bacterial_wilt*)
- **Чёрная парша** (*Potato_black_scurf*)
- **Чёрная ножка** (*Potato_black_shank_disease*)
- **Обыкновенная язва** (*Potato_common_scab*)
- **Сухая гниль** (*Potato_dry_rot*)
- **Ранний фитофтороз** (*Potato_early_blight*)
- **Поздний фитофтороз** (*Potato_late_blight*)
- **Вертициллезное увядание** (*Potato_verticillium_wilt*)
- **Бородавчатая болезнь** (*Potato_wart_disease*)
- **Увядание** (*Potato_wilt*)
- **Порошковая язва** (*Powdery_scab_of_potato*)
- **Кольцевая бактериальная гниль** (*potato_bacterial_ring_rot*)

### Заболевания листьев
- **Здоровый лист** (*Healthy*)
- **Ранний фитофтороз** (*Early_Blight*)
- **Поздний фитофтороз** (*Late_Blight*)

---
## Метаданные

### Клубни (`tuber_metadata.json`)
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

### Листья (`leaves_metadata.json`)
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

---
## Установка и запуск

### 1️ Клонирование репозитория
```sh
git clone https://github.com/niyaz121221324/PotatoDisease.API.git
cd PotatoDisease.API
```

### 2️ Запуск через Docker
```sh
docker-compose up --build
```

---
## Использование API

### Отправка изображения для анализа
Отправьте **POST**-запрос на `/predict` с изображением картофеля:
```sh
curl -X POST "http://localhost:8080/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/image.jpg"
```

### Ответ сервера
Ответ содержит предсказанный класс и вероятность:
```json
{
    "predictedClass": "Potato___Early_blight",
    "confidence": 0.95
}
```

---
## 🛠 Технологии
✅ **FastAPI** — создание REST API  
✅ **TensorFlow** — машинное обучение  
✅ **Docker** — контейнеризация  
✅ **Python** — основной язык разработки  

Проект разработан для анализа качества картофеля и раннего выявления его заболеваний.