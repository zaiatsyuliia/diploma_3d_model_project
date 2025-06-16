# 3D Модель з 2D Зображення
Веб-додаток для створення 3D моделей з 2D зображень за допомогою глибокого навчання.

## Встановлення
1. Клонуйте репозиторій:
```
git clone https://github.com/zaiatsyuliia/diploma_3d_model_project.git
```
2. Встановіть залежності:
```
pip install -r requirements.txt
```
3. Запустіть сервер:
```
python server.py
```
4. Відкрийте у браузері: http://localhost:8000

## Використання
1. Завантажте зображення

   ![image](https://github.com/user-attachments/assets/ab0095d4-5590-4264-8db1-74c53b68265d)
2. Налаштуйте параметри (товщина, модель)

   ![image](https://github.com/user-attachments/assets/bd335358-fddd-4b71-a79e-c911bc5cdaa6)
3. Натисніть "3D-модель" для генерації та "Карта глибин" для показу глибини зображення

   ![image](https://github.com/user-attachments/assets/cd09b38d-e7ab-4f35-9b57-c5614c9aba02)
4. Перегляд результату

   ![image](https://github.com/user-attachments/assets/d274e44b-cf9c-4516-8c51-9159f1ec4302)
   ![image](https://github.com/user-attachments/assets/b302512c-5b8c-4e6f-a7fd-b3f95c6da3eb)
5. Завантажте OBJ файл

   ![image](https://github.com/user-attachments/assets/7ab7d65b-985a-4128-9e1f-3dc8252a44f8)

## Структура проєкту
```
├── server.py              # Flask сервер
├── depth_map.py           # Генерація карт глибини
├── create_3dmodel.py      # Створення 3D моделей
├── templates/
│   └── create_model.html  # Веб-інтерфейс
├── static/
│   └── style.css         # Стилі
└── requirements.txt       # Залежності
```

## Можливості
- Генерація 3D моделей з звичайних 2D фотографій
- Автоматичне видалення фону
- Створення карт глибини
- 3D-перегляд у браузері
- Експорт в формат OBJ

## Моделі глибини
- DPT Large - найкраща якість
- DPT Hybrid - компроміс
- MiDaS Small - швидкість
