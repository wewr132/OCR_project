# src/api.py
"""
Модуль взаимодействия с Google Gemini API.
Поддерживает два режима: извлечение метаданных (JSON) и тела документа (текст).
"""

import os
import json
from pathlib import Path
from typing import Optional
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

class GeminiClient:
    """Клиент для работы с мультимодальной моделью Gemini 2.0 Flash."""
    
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """Инициализация клиента."""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
    
    def _get_header_prompt(self) -> str:
        """Системный промпт для извлечения метаданных из шапки документа."""
        return """Ты — помощник для извлечения метаданных из официальных документов РФ.
На изображении — верхняя часть документа (шапка). Твоя задача:
1. Найти и извлечь НОМЕР документа (формат: цифры, возможно с буквами/дефисами, например: "123-п", "№ 45/2025").
2. Найти и извлечь ДАТУ документа (привести к формату ГГГГ-ММ-ДД, если возможно).
3. Определить ТИП документа (например: "Приказ", "Распоряжение", "Письмо").
4. Извлечь НАИМЕНОВАНИЕ организации-отправителя, если оно есть.

Требования к ответу:
- Верни ТОЛЬКО валидный JSON без дополнительных пояснений.
- Используй ключи: "doc_number", "doc_date", "doc_type", "issuer".
- Если поле не найдено — укажи null.
- Не выдумывай значения. Если текст неразборчив — лучше null.

Пример ответа:
{
  "doc_number": "123-п",
  "doc_date": "2025-09-01",
  "doc_type": "Приказ",
  "issuer": "Арбитражный суд Республики Карелия"
}"""

    def _get_body_prompt(self) -> str:
        """Системный промпт для извлечения основного текста документа."""
        return """Ты — система оцифровки официальных документов. Твоя задача:
1. Перепиши ВЕСЬ печатный текст с изображения, сохраняя структуру абзацев и нумерацию пунктов.
2. Игнорируй рукописные подписи в конце документа. Если видишь подпись — оставь маркер [ПОДПИСЬ] и не пытайся её расшифровать.
3. Игнорируй печати, штампы, водяные знаки и другие графические элементы.
4. Если встречаются неразборчивые фрагменты печатного текста — оставь [?] на их месте.
5. Не добавляй комментариев, пояснений или форматирования вне текста документа.

Важно: возвращай только чистый текст документа, без markdown, без кавычек, без преамбул."""

    def extract_metadata(self, image_path: Path) -> Optional[dict]:
        """
        Извлекает метаданные из изображения шапки документа.
        Возвращает словарь или None при ошибке.
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Ошибка: файл не найден {image_path}")
            return None
        
        try:
            # Загружаем изображение
            image = genai.upload_file(str(image_path))
            
            # Генерируем ответ с требованием JSON
            response = self.model.generate_content(
                contents=[self._get_header_prompt(), image],
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.1,  # Минимальная креативность для точности
                )
            )
            
            # Парсим JSON
            result = json.loads(response.text.strip())
            return result
            
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON от API: {e}")
            print(f"Полученный ответ: {response.text[:200]}...")
            return None
        except Exception as e:
            print(f"Ошибка при извлечении метаданных: {e}")
            return None
        finally:
            # Освобождаем ресурс, если используется временная загрузка
            if 'image' in locals():
                try:
                    image.delete()
                except:
                    pass

    def extract_body_text(self, image_path: Path) -> Optional[str]:
        """
        Извлекает основной текст из изображения тела документа.
        Возвращает строку или None при ошибке.
        """
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        if not image_path.exists():
            print(f"Ошибка: файл не найден {image_path}")
            return None
        
        try:
            image = genai.upload_file(str(image_path))
            
            # Генерируем ответ в текстовом формате (без JSON)
            response = self.model.generate_content(
                contents=[self._get_body_prompt(), image],
                generation_config=GenerationConfig(
                    temperature=0.2,  # Чуть выше для естественности текста
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Ошибка при извлечении текста тела: {e}")
            return None
        finally:
            if 'image' in locals():
                try:
                    image.delete()
                except:
                    pass