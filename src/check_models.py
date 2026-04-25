from google import genai
from config import Config

# Загружаем настройки (ключ и прокси подхватятся автоматически)
Config.validate()

client = genai.Client(api_key=Config.GEMINI_API_KEY)

print("\n--- Доступные модели для генерации контента ---")
for model in client.models.list():
    # Нас интересуют только модели, поддерживающие генерацию
    if 'generateContent' in model.supported_actions:
        print(model.name)
print("-----------------------------------------------")