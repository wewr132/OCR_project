import base64
import json
import requests
from pathlib import Path
from config import Config

class YandexVLMClient:
    """
    Адаптер для взаимодействия с Yandex Cloud Foundation Models.
    Поддерживает мультимодальные модели (Vision).
    """
    def __init__(self):
        # Используем API-ключ сервисного аккаунта для долгосрочной работы без ротации IAM-токенов
        self.api_key = Config.YANDEX_API_KEY
        self.folder_id = Config.YANDEX_FOLDER_ID
        self.model_uri = f"vis://{self.folder_id}/qwen-3.5-35b-vision/latest" 
        self.headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json"
        }
        self.endpoint = "https://llm.api.cloud.yandex.net/foundationModels/v1/chat"

    def _encode_image(self, image_path: Path) -> str:
        """Кодирует изображение в Base64 для передачи в JSON."""
        with open(image_path, "rb") as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded

    def _get_header_prompt(self) -> str:
        return """
        Ты — автоматизированная система извлечения метаданных из официальных приказов.
        Твоя задача: найти дату приказа и его регистрационный номер.
        Особое внимание обращай на рукописные символы.
        Верни результат СТРОГО в формате JSON без markdown-разметки:
        {"date": "найденная дата", "number": "найденный номер"}
        Если данные не найдены, верни null. Не пиши ничего, кроме JSON.
        """

    def _get_body_prompt(self) -> str:
        return """
        Ты — система оцифровки официальных документов. Перепиши весь печатный текст с предоставленного изображения. 
        Соблюдай следующие жесткие правила:
        1. Игнорируй любые рукописные подписи в конце документа.
        2. Игнорируй круглые печати и угловые штампы.
        3. Сохраняй исходное разбиение на абзацы.
        4. Не добавляй от себя никаких комментариев.
        """

    def _send_request(self, image_path: Path, prompt: str) -> str | None:
        """Формирует payload и выполняет POST-запрос к API."""
        if not image_path.exists():
            print(f"Ошибка: файл {image_path} не найден.")
            return None

        base64_img = self._encode_image(image_path)
        
        payload = {
            "modelUri": self.model_uri,
            "messages": [
                {
                    "role": "system",
                    "text": prompt
                },
                {
                    "role": "user",
                    "text": "Обработай это изображение согласно инструкциям.",
                    "image": base64_img
                }
            ],
            "completionOptions": {
                "stream": False,
                "temperature": 0.1, # Низкая температура для воображения модели
                "maxTokens": 2000
            }
        }

        try:
            response = requests.post(self.endpoint, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result['result']['alternatives'][0]['message']['text']
        except Exception as e:
            print(f"Ошибка обращения к API Yandex: {e}")
            return None

    def extract_metadata(self, image_path: Path) -> dict | None:
        """Извлечение шапки. Возвращает словарь."""
        if Config.USE_MOCK_API:
            return {"date": "01.09.2025", "number": "141-П"}

        raw_response = self._send_request(image_path, self._get_header_prompt())
        if not raw_response:
            return None
            
        try:
            # Очистка возможных артефактов модели перед парсингом
            clean_text = raw_response.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            print(f"Ошибка парсинга JSON. Сырой ответ: {raw_response}")
            return None

    def extract_body_text(self, image_path: Path) -> str | None:
        """Извлечение текста тела документа. Возвращает строку."""
        if Config.USE_MOCK_API:
            return "Тестовый текст тела документа из Mock-режима."

        return self._send_request(image_path, self._get_body_prompt())