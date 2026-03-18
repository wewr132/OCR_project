import cv2
import numpy as np
from pathlib import Path
from config import Config

class ImagePreprocessor:
    """
    Класс для подготовки изображений к распознаванию.
    Выполняет сегментацию (кроп) и адаптивную бинаризацию.
    """

    @staticmethod
    def process_document(image_path: Path) -> Path | None:
        # Убеждаемся, что на вход пришел объект Path
        if isinstance(image_path, str):
            image_path = Path(image_path)

        if not image_path.exists():
            print(f"Ошибка: Файл не найден {image_path}")
            return None

        # 1. Загрузка в Grayscale. Передаем строку, так как cv2 не всегда дружит с Path
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Ошибка чтения изображения (возможно, поврежден файл): {image_path.name}")
            return None

        # 2. Размеры и кроп
        height, width = image.shape[:2]
        crop_h = int(height * Config.HEADER_HEIGHT)
        head_img = image[0:crop_h, 0:width]
        
        # 3. Предобработка (Удаление шума и адаптивный порог)
        # Ядро (5, 5) уберет мелкую грязь со скана
        blurred = cv2.GaussianBlur(head_img, (5, 5), 0)
        
        # Адаптивная бинаризация: размер блока 11, константа вычитания 2 (подбирается экспериментально)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # 4. Сохранение
        processed_path = image_path.parent / f"proc_{image_path.name}"
        
        success = cv2.imwrite(str(processed_path), thresh)
        
        if success:
            return processed_path
        else:
            print(f"Ошибка при сохранении: {processed_path}")
            return None

if __name__ == "__main__":
    print("Запуск препроцессора...")
    # Обязательно передаем объект Path
    test_file = Path('/home/bogdan/inf/OCR_project/data/output/test/page_1.png')
    result = ImagePreprocessor.process_document(test_file)
    print(f'Результат сохранен по пути: {result}')