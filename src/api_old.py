# src/api.py
import json
import base64
import time
import requests
from pathlib import Path
from typing import Optional
from config import Config

class YandexClient:
    """
    Клиент для работы с Yandex Cloud.
    Использует YandexGPT Vision для шапки (JSON) и Yandex Vision OCR для тела (Текст).
    """
    
    def __init__(self):
        # Безопасное получение доступов из Config (чтобы не падало в Mock-режиме)
        self.folder_id = getattr(Config, 'YANDEX_FOLDER_ID', 'dummy_folder_id')
        self.api_key = getattr(Config, 'YANDEX_API_KEY', 'dummy_api_key')
        
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # эндпоинты Яндекса
        self.ocr_url = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
        self.gpt_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        
        
    def _encode_image(self, image_path: Path) -> str:
        """Кодирует изображение в Base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _retry_request(self, func, max_retries=3):
        """Экспоненциальная задержка при лимитах (429 ошибка)."""
        for i in range(max_retries):
            try:
                return func()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    wait = (2 ** i) * 5
                    print(f"Лимит Yandex API. Жду {wait}с...")
                    time.sleep(wait)
                else:
                    print(f"Ошибка HTTP: {e.response.text}")
                    raise
        return None

    def extract_metadata(self, image_path: Path) -> Optional[dict]:
        """Извлекает метаданные через YandexGPT Vision (структурированный JSON)."""
        
        # --- ВАЖНО: Возврат фейковых данных для локального тестирования ---
        if getattr(Config, 'USE_MOCK_API', False):
            print("🤖 [MOCK] YandexGPT Vision: Возврат фейковых метаданных")
            return {
                "doc_number": "141-П", 
                "doc_date": "2025-09-01", 
                "doc_type": "Приказ", 
                "issuer": "Арбитражный суд"
            }

        img_b64 = self._encode_image(image_path)
        prompt = """Ты — помощник для извлечения метаданных из официальных документов РФ.
На изображении — шапка документа. Обрати внимание, что на фото есть рукописные цифры и буквы. Найди:
1. Номер документа (ключ: "doc_number")
2. Дату документа в формате ГГГГ-ММ-ДД (ключ: "doc_date")
3. Тип документа (ключ: "doc_type")
4. Организацию-отправителя (ключ: "issuer")

Верни ТОЛЬКО валидный JSON без markdown-обёрток и пояснений. Если поле не найдено — укажи null."""

        payload = {
            "modelUri": f"vis://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False, 
                "temperature": 0.1, 
                "maxTokens": 1000
            },
            "messages": [
                {
                    "role": "system",
                    "text": prompt
                },
                {
                    "role": "user",
                    "text": "Извлеки данные из изображения",
                    "image": img_b64
                }
            ]
        }

        def _call():
            resp = requests.post(self.vlm_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry_request(_call)
            if not result: return None
            
            text = result["result"]["alternatives"][0]["message"]["text"]
            # Очистка JSON от возможных тегов, которые иногда генерирует ИИ
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
            
        except Exception as e:
            print(f"❌ Ошибка YandexGPT (метаданные): {e}")
            return None

    def extract_body_text(self, image_path: Path) -> Optional[str]:
        """Извлекает основной текст через классический Yandex Vision OCR."""
        
        # --- Возврат фейковых данных для локального тестирования ---
        if getattr(Config, 'USE_MOCK_API', False):
            print("[MOCK] Yandex Vision OCR: Возврат фейкового текста")
            return "Тестовый текст тела документа.\nСидоров Ю.В. назначен ответственным.\nПодпись."

        img_b64 = self._encode_image(image_path)
        
        # Специфичный формат payload для Yandex Vision OCR
        payload = {
            "folderId": self.folder_id,
            "analyze_specs": [{
                "content": img_b64,
                "features": [{
                    "type": "TEXT_DETECTION",
                    "text_detection_config": {
                        "language_codes": ["ru"]
                    }
                }]
            }]
        }

        def _call():
            resp = requests.post(self.ocr_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry_request(_call)
            if not result: return None
            
            # Разбор сложного JSON-ответа от Yandex Vision
            texts = []
            try:
                pages = result['results'][0]['results'][0]['textDetection']['pages']
                for page in pages:
                    for block in page.get('blocks', []):
                        for line in block.get('lines', []):
                            # Собираем слова в одну строку
                            line_text = " ".join([w.get('text', '') for w in line.get('words', [])])
                            if line_text:
                                texts.append(line_text)
                return "\n".join(texts)
            except (KeyError, IndexError):
                print("Текст на изображении не найден или структура ответа пуста.")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка Yandex Vision (текст): {e}")
            return None