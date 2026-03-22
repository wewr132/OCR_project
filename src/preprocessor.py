import cv2
import numpy as np
from pathlib import Path
from config import Config

class ImagePreprocessor:
    """
    Класс для подготовки изображений.
    Логика: валидация -> нарезка сырого массива -> раздельная обработка.
    """

    @classmethod
    def _validate_path(cls, path_input: Path | str) -> Path | None:
        path_obj = Path(path_input)
        if not path_obj.exists():
            print(f"Ошибка: Файл не найден {path_obj}")
            return None
        return path_obj

    @classmethod
    def process_document(cls, image_path: Path | str) -> tuple[Path, Path] | None:
        """
        Единая точка входа. Нарезает изображение и применяет фильтры
        целевым образом.
        """
        valid_path = cls._validate_path(image_path)
        if not valid_path: return None

        # 1. Загрузка в Grayscale
        image = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Ошибка чтения: {valid_path.name}")
            return None

        # 2. Вычисление координат сечения
        height, width = image.shape[:2]
        crop_h = int(height * Config.HEADER_HEIGHT)
        
        # 3. Срезы массива
        head_img = image[0:crop_h, 0:width]
        body_img = image[crop_h:height, 0:width]

        # 4. Применяем фильтры ТОЛЬКО к шапке (для ИИ)
        blurred = cv2.GaussianBlur(head_img, (5, 5), 0)
        head_processed = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )

        # 5. Формируем пути
        head_path = valid_path.parent / f"proc_head_{valid_path.name}"
        body_path = valid_path.parent / f"raw_body_{valid_path.name}" # raw - сырой
        
        # 6. Сохраняем файлы
        success_h = cv2.imwrite(str(head_path), head_processed)
        success_b = cv2.imwrite(str(body_path), body_img)

        if success_h and success_b:
            return (head_path, body_path)
        else:
            print("Ошибка записи на диск.")
            return None

if __name__ == "__main__":
    test_file = Path('/home/bogdan/OCR_project/data/output/test/page_1.png')
    result = ImagePreprocessor.process_document(test_file)
    if result:
        print(f"Обработка завершена.\nШапка для ИИ: {result[0]}\nТело для Tesseract: {result[1]}")