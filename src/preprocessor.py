# src/preprocessor.py
import cv2
import numpy as np
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentPreprocessor:
    """
    Предобработка изображений для Vision LLM.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_header(self, src_path: Path, ratio: float = 0.25) -> Path:
        """Вырезает верхнюю часть изображения (шапку) для извлечения метаданных."""
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось прочитать: {src_path}")
            
        h, w = img.shape[:2]
        crop_h = int(h * ratio)
        header = img[0:crop_h, 0:w]
        
        binary = cv2.adaptiveThreshold(
            header, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )

        # 3. Сохранение в JPEG (как мы и договорились)
        header_path = self.output_dir /  f"header_{src_path.name}"
        cv2.imwrite(str(header_path), binary, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return header_path

    def _enhance_for_vlm(self, image: np.ndarray) -> np.ndarray:
        # 1. Мягкое подавление шума
        denoised = cv2.fastNlMeansDenoising(image, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. Локальное улучшение контраста
        # Делает бледные чернила четче
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def preprocess_single(self, src_path: Path) -> Path:
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Не удалось прочитать изображение: {src_path}")

        processed = self._enhance_for_vlm(img)
        
        out_path = self.output_dir / f"proc_{src_path.name}"
        if not cv2.imwrite(str(out_path), processed):
            raise RuntimeError(f"Ошибка сохранения: {out_path}")
            
        return out_path

    def preprocess_batch(self, src_paths: List[Path]) -> List[Path]:
        """Обрабатывает пакет изображений, возвращает список путей к новым файлам."""
        processed = []
        for p in src_paths:
            try:
                processed.append(self.preprocess_single(p))
            except Exception as e:
                logger.warning(f"Пропуск {p.name}: {e}")
        return processed

    @staticmethod
    def cleanup(paths: List[Path], strict: bool = False) -> int:
        """Удаляет временные JPEG после успешной записи в БД."""
        deleted = 0
        for p in paths:
            if p.exists():
                try:
                    p.unlink()
                    deleted += 1
                except Exception as e:
                    if strict:
                        logger.error(f"Не удалось удалить {p}: {e}")
                        raise e
        logger.info(f"Удалено временных файлов: {deleted}")
        return deleted