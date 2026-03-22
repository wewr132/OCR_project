import pytesseract
from pathlib import Path
import cv2

class LocalOCR:
    """
    Класс для извлечения печатного текста из тела документа.
    Использует локальный движок Tesseract OCR.
    """

    @staticmethod
    def extract_text(image_path: Path) -> str | None:
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            print(f"Ошибка: Файл тела документа не найден {image_path}")
            return None

        try:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            
            text = pytesseract.image_to_string(image, lang='rus', config ='--psm 1').strip()
            text = text.replace('©', '').replace('— —', '—')
            
            return text

        except pytesseract.TesseractNotFoundError:
            print("Критическая ошибка: Tesseract не установлен в системе.")
            print("Выполните: sudo pacman -S tesseract tesseract-data-rus")
            return None

        except Exception as e:
            print(f"Ошибка при распознавании текста локально: {e}")
            return None

if __name__ == "__main__":
    # Тест
    # Передай путь к сгенерированному файлу proc_body_page_1.png
    test_file = Path('/home/bogdan/OCR_project/data/output/test/raw_body_page_1.png')
    result = LocalOCR.extract_text(test_file)
    print("--- Извлеченный текст ---")
    print(result)