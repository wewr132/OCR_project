# src/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# === 1. Загрузка переменных окружения ===
load_dotenv()

# === 2. Глобальные переменные (доступны напрямую) ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "Ошибка конфигурации: GEMINI_API_KEY не найден!\n"
        "Убедись, что в корне проекта есть файл .env с строкой:\n"
        "GEMINI_API_KEY=ключ_без_кавычек"
    )

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "input"
OUTPUT_DIR = BASE_DIR / "data" / "output"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# === 3. Класс-контейнер для настроек (опционально, для удобства) ===
class Config:
    """Контейнер для настроек приложения. Дублирует глобальные переменные для удобного доступа."""
    
    # API
    GEMINI_API_KEY = GEMINI_API_KEY  # <-- Дублируем, чтобы работало cls.GEMINI_API_KEY
    MODEL_NAME = "gemini-1.5-flash-8b"
    PROXY_URL = os.getenv("PROXY_URL", "http://127.0.0.1:10809")
    
    # Обработка изображений
    DPI = 300
    HEADER_HEIGHT = 0.25
    
    # Пути (дублируем для доступа через класс)
    INPUT_DIR = INPUT_DIR
    OUTPUT_DIR = OUTPUT_DIR
    PROCESSED_DIR = PROCESSED_DIR

    @classmethod
    def validate(cls):
        """Проверка конфигурации и создание папок."""
        if not cls.GEMINI_API_KEY:
            raise ValueError("Критическая ошибка: GEMINI_API_KEY не найден!")
        
        # Настройка прокси
        if cls.PROXY_URL:
            os.environ['HTTP_PROXY'] = cls.PROXY_URL
            os.environ['HTTPS_PROXY'] = cls.PROXY_URL
            print(f"Прокси: {cls.PROXY_URL}")
        
        # Создание папок
        for dir_path in [cls.INPUT_DIR, cls.OUTPUT_DIR, cls.PROCESSED_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Готово: {dir_path}")
        
        return True


# === 4. ПРОВЕРКА ЗАПУСКА МОДУЛЯ ===
if __name__ == "__main__":
    print("---> Запуск проверки config.py...\n")
    
    try:
        # 1. Проверка глобальных переменных
        print(" Глобальные переменные:")
        print(f"   GEMINI_API_KEY: {GEMINI_API_KEY[:10] + '...' if GEMINI_API_KEY else 'ПУСТО'}")
        print(f"   BASE_DIR: {BASE_DIR}")
        
        # 2. Проверка класса Config
        print("\n Класс Config:")
        print(f"   MODEL_NAME: {Config.MODEL_NAME}")
        print(f"   DPI: {Config.DPI}")
        
        # 3. Запуск валидации (создание папок)
        print("\n---> Запуск валидации...")
        if Config.validate():
            print("\nКонфигурация загружена успешно")
            print("Готовые пути:")
            print(f"   INPUT:      {Config.INPUT_DIR}")
            print(f"   OUTPUT:     {Config.OUTPUT_DIR}")
            print(f"   PROCESSED:  {Config.PROCESSED_DIR}")
            
    except Exception as e:
        print(f"\nОшибка при проверке конфигурации:\n{e}")
        exit(1)