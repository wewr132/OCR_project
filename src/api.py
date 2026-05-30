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
    Использует каскад: Vision OCR (handwritten) -> YandexGPT (Text) для шапки
    и Vision OCR (page) с пространственной фильтрацией для тела.
    """
    
    def __init__(self):
        self.folder_id = getattr(Config, 'YANDEX_FOLDER_ID', 'dummy_folder_id')
        self.api_key = getattr(Config, 'YANDEX_API_KEY', 'dummy_api_key')
        
        # Единый правильный HTTP-заголовок для статических API-ключей.
        # Схема Api-Key строго исключает использование заголовка x-folder-id.
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Эндпоинты Яндекса
        self.ocr_url = "https://vision.api.cloud.yandex.net/vision/v1/batchAnalyze"
        self.gpt_url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
        self.ocr_recognize_url = "https://ocr.api.cloud.yandex.net/ocr/v1/recognizeText"

    def _encode_image(self, image_path: Path) -> str:
        """Кодирует изображение в Base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _retry_request(self, func, max_retries=3):
        """Экспоненциальная задержка при лимитах с логированием HTTP-ошибок."""
        for i in range(max_retries):
            try:
                return func()
            except requests.exceptions.HTTPError as e:
                status = e.response.status_code
                if status == 429:
                    wait = (2 ** i) * 5
                    print(f"⚠️ Лимит Yandex API (429). Жду {wait}с...")
                    time.sleep(wait)
                else:
                    print(f"❌ Ошибка HTTP {status}: {e.response.text}")
                    raise
            except requests.exceptions.RequestException as e:
                print(f"❌ Ошибка сети: {e}")
                raise
        return None

    # --- ШАГ 1: ЧТЕНИЕ ПОЧЕРКА ИЗ ШАПКИ ---
    def extract_header_raw_text(self, image_path: Path) -> Optional[str]:
        """Извлекает сырой текст из обрезанной шапки (модель handwritten через новый API)."""
        if getattr(Config, 'USE_MOCK_API', False):
            print("🤖 [MOCK] Yandex Vision OCR: Возврат сырого текста шапки")
            return "Приказ № 141-П от 2025-09-01 Арбитражный суд"

        img_b64 = self._encode_image(image_path)
        
        # Плоская структура JSON согласно спецификации recognizeText
        payload = {
            "mimeType": "JPEG",
            "languageCodes": ["ru"],
            "model": "handwritten",
            "content": img_b64
        }

        def _call():
            # Используем корректный единый заголовок self.headers
            resp = requests.post(self.ocr_recognize_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry_request(_call)
            if not result: 
                return None
            
            # Десериализация графа: поиск textAnnotation
            text_annotation = result.get('textAnnotation') or result.get('result', {}).get('textAnnotation', {})
            
            if not text_annotation:
                print(f"⚠️ [DEBUG] Структура 'textAnnotation' не найдена. Ответ API: {json.dumps(result, ensure_ascii=False)}")
                return None

            extracted_lines = []
            blocks = text_annotation.get('blocks', [])
            
            for block in blocks:
                for line in block.get('lines', []):
                    # Прямое извлечение текста на уровне строки
                    line_text = line.get('text', '').strip()
                    if line_text:
                        extracted_lines.append(line_text)
                    else:
                        # Резервный обход лексем
                        words = [w.get('text', '') for w in line.get('words', [])]
                        if words:
                            extracted_lines.append(" ".join(words).strip())
                            
            final_text = " ".join(extracted_lines).strip()
            if not final_text:
                print(f"⚠️ [DEBUG] Текст распознан как пустой. Ответ API: {json.dumps(result, ensure_ascii=False)}")
                return None
                
            return final_text
            
        except Exception as e:
            print(f"❌ Сбой в extract_header_raw_text: {e}")
            return None

    # --- ШАГ 2: ПАРСИНГ ТЕКСТА ШАПКИ (YANDEX GPT) ---
    def extract_metadata(self, clean_header_text: str) -> Optional[dict]:
        """Извлекает метаданные из текста через YandexGPT."""
        if getattr(Config, 'USE_MOCK_API', False):
            print("🤖 [MOCK] YandexGPT: Возврат фейковых метаданных")
            return {"doc_number": "141-П", "doc_date": "2025-09-01", "doc_type": "Приказ", "issuer": "Арбитражный суд"}

        if not clean_header_text:
            return None

        prompt = """Ты — помощник для извлечения метаданных из официальных документов РФ.
В тебе передан распознанный текст шапки документа. Найди:
1. Номер документа (ключ: "doc_number")
2. Дату документа в формате ГГГГ-ММ-ДД (ключ: "doc_date")
3. Тип документа (ключ: "doc_type")
4. Организацию-отправителя (ключ: "issuer")

Верни ТОЛЬКО валидный JSON без markdown-обёрток и пояснений. Если поле не найдено — укажи null."""

        payload = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {
                "stream": False, 
                "temperature": 0.0,
                "maxTokens": 1000
            },
            "messages": [
                {"role": "system", "text": prompt},
                {"role": "user", "text": clean_header_text}
            ]
        }

        def _call():
            resp = requests.post(self.gpt_url, headers=self.headers, json=payload)
            resp.raise_for_status()
            return resp.json()

        try:
            result = self._retry_request(_call)
            if not result: 
                return None
            
            text = result["result"]["alternatives"][0]["message"]["text"]
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
            
        except Exception as e:
            print(f"❌ Ошибка YandexGPT (метаданные): {e}")
            return None

    # --- ШАГ 3: ЧТЕНИЕ ТЕЛА С ФИЛЬТРАЦИЕЙ ---
    def extract_body_text(self, image_path: Path, crop_y_threshold: int = 0) -> Optional[str]:
        """Извлекает основной текст, отсекая мусор из шапки по координатам."""
        if getattr(Config, 'USE_MOCK_API', False):
            print("[MOCK] Yandex Vision OCR: Возврат фейкового текста тела")
            return "Тестовый текст тела документа.\nСидоров Ю.В. назначен ответственным.\nПодпись."

        img_b64 = self._encode_image(image_path)
        
        payload = {
            "folderId": self.folder_id,
            "analyze_specs": [{
                "content": img_b64,
                "features": [{
                    "type": "TEXT_DETECTION",
                    "text_detection_config": {
                        "language_codes": ["ru"],
                        "model": "page"
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
            if not result: 
                return None
            
            body_lines = []
            try:
                pages = result['results'][0]['results'][0]['textDetection']['pages']
                for page in pages:
                    for block in page.get('blocks', []):
                        for line in block.get('lines', []):
                            words = line.get('words', [])
                            if not words: 
                                continue
                            
                            try:
                                y_coords = [int(float(w['boundingBox']['vertices'][0].get('y', 0))) for w in words]
                                avg_y = sum(y_coords) / len(y_coords)
                            except (KeyError, IndexError):
                                continue

                            if avg_y >= crop_y_threshold:
                                line_text = " ".join([w.get('text', '') for w in words])
                                if line_text.strip():
                                    body_lines.append(line_text.strip())
                                    
                return "\n".join(body_lines)
            except (KeyError, IndexError):
                print("Текст на изображении не найден или структура ответа пуста.")
                return None
                
        except Exception as e:
            print(f"Ошибка Yandex Vision (текст тела): {e}")
            return None