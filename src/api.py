import json
from pathlib import Path
from google import genai
from google.genai import types
from config import Config

class GeminiClient:
    """
    Класс для взаимодействия с Google Gemini API.
    Использует актуальный SDK (google-genai).
    """
    def __init__(self):
        # Инициализация клиента с ключом из конфига
        self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
        self.model_name = Config.MODEL_NAME

    def _get_system_prompt(self) -> str:
        return """
        Ты — автоматизированная система извлечения метаданных из официальных приказов.
        Тебе на вход подается верхняя часть документа (шапка).
        Твоя задача: найти дату приказа и его регистрационный номер.
        Особое внимание обращай на рукописные символы.
        Верни результат СТРОГО в формате JSON без markdown-разметки:
        {"дата": "найденная дата", "номер": "найденный номер"}
        Если данные не найдены, верни null.
        """

    def extract_metadata(self, image_path: Path | str) -> dict | None:
        # Универсальная проверка пути
        path_obj = Path(image_path)
        if not path_obj.exists():
            print(f"Ошибка: Файл не найден {path_obj}")
            return None

        uploaded_file = None
        try:
            # 1. Загрузка файла на сервер
            print(f"Загрузка файла {path_obj.name} на сервер...")
            uploaded_file = self.client.files.upload(file=str(path_obj))
            
            # 2. Формирование конфигурации
            # Явно указываем модели формат ожидаемого ответа
            config = types.GenerateContentConfig(
                system_instruction=self._get_system_prompt(),
                response_mime_type="application/json",
            )
            
            # 3. Вызов генерации
            print("Ожидание ответа от нейросети...")
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[uploaded_file],
                config=config
            )
            
            # 4. Парсинг ответа
            raw_text = response.text.strip()
            
            # На случай, если модель проигнорировала запрет на markdown
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:-3].strip()
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:-3].strip()

            return json.loads(raw_text)
            
        except json.JSONDecodeError as e:
            print(f"Ошибка парсинга JSON: {e}\nСырой ответ: {raw_text}")
            return None
        except Exception as e:
            print(f"Критическая ошибка API: {e}")
            return None
        finally:
            # 5. Гарантированное удаление файла
            # Блок finally выполняется всегда, даже если в try произошла ошибка или return
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    print("Временный файл удален с сервера.")
                except Exception as e:
                    print(f"Не удалось удалить файл: {e}")

if __name__ == "__main__":
    Config.validate()
    test_file = Path('/home/bogdan/OCR_project/data/output/test/proc_head_page_1.png')
    
    # ПРАВИЛЬНЫЙ ВЫЗОВ:
    # 1. Создаем экземпляр (объект) класса
    api_client = GeminiClient()
    
    # 2. Вызываем метод у созданного экземпляра
    result = api_client.extract_metadata(test_file)
    
    if result:
        print(f"Обработка завершена. Результат:\n{result}")
    else:
        print('Ошибка извлечения данных.')