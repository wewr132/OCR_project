import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Загружаем переменные из .env
load_dotenv()

# 2. Определяем базовый путь к проекту (корневая папка)
BASE_DIR = Path(__file__).resolve().parent.parent

class Config:
    # API Настройки
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    MODEL_NAME = "gemini-1.5-flash" 

    # Пути к данным
    INPUT_DIR = BASE_DIR / "data" / "input"
    OUTPUT_DIR = BASE_DIR / "data" / "output"

    # Настройки обработки изображений (для OpenCV)
    DPI = 300            # Высокое разрешение для мелкого рукописного текста [cite: 84]
    HEADER_HEIGHT = 0.3  # Берем верхние 30% страницы для сегментации "шапки" 

    @classmethod
    def validate(cls):
        """Проверка, что все настройки на месте"""
        if not cls.GEMINI_API_KEY:
            raise ValueError("Критическая ошибка: GEMINI_API_KEY не найден в .env!")
        
        proxy = os.getenv("PROXY_URL")
        if proxy:
            os.environ['HTTP_PROXY'] = proxy
            os.environ['HTTPS_PROXY'] = proxy
            print(f"Включена маршрутизация трафика через прокси: {proxy}")
        # Создаем папки, если их нет
        cls.INPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print("Конфигурация успешно загружена.")