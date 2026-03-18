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
        # 1. Загрузи изображение через cv2.imread
        image = cv2.imread(f"{image_path}", cv2.IMREAD_GRAYSCALE)
        # 2. Получи размеры (высоту и ширину)
        if image is None:
            print(f"ошибка чтения cv2.imread {image_path.name}")
        else:
            # 2. Получи размеры (высоту и ширину)
            (height, width) = image.shape[:2]
            print()
            # 3. Сделай кроп "шапки"
            head_img = image[0:int(height * Config.HEADER_HEIGHT), 0:width]
        
            
            blurred_image = cv2.GaussianBlur(head_img, (5, 5), 0)

            thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
            image_path = Path(image_path)
            # 6. Сформируй путь для сохранения (например, в ту же папку с префиксом 'proc_')
            processed_path = image_path.parent / f"proc_{image_path.name}"
            processed_path.touch(exist_ok=True)
            # 7. Сохрани результат через cv2.imwrite
            succes = cv2.imwrite(processed_path, thresh)
            if succes is not False:
                return processed_path
            else:
                print(f'Ошибка при сохранении: {processed_path}')

if __name__ == "__main__":
    print("start")
    result = ImagePreprocessor.process_document('/home/bogdan/inf/OCR_project/data/output/test/page_1.png')
    print(f'result: {result}')