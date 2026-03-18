import os
from pathlib import Path
from pdf2image import convert_from_path
from config import Config

class PDFConverter:
    @staticmethod
    def convert(pdf_name: str) -> list[Path]:
        pdf_path = Config.INPUT_DIR / pdf_name # абсолютный путь до файла
        output_subdir = Config.OUTPUT_DIR / Path(pdf_name).stem
        output_subdir.mkdir(parents=True, exist_ok=True) #подпапка для png страний файла

        if not pdf_path.exists():
            print(f'ОШИБКА! файл {pdf_name} не найден.')
            return []

        image_path = []   
        try:
            print(f'Растеризация: {pdf_name}...')
            images = convert_from_path(pdf_path, dpi=Config.DPI)
            for i, image in enumerate(images):
                # даём имя странице и пишем ее путь
                image_name = f'page_{i+1}.png'
                full_image_path = output_subdir / image_name
                # сохраняем объект по пути
                image.save(full_image_path, "PNG")
                image_path.append(full_image_path)
                print(f'Coхранено: {full_image_path.name}')
            return image_path
        

        except Exception as e:
            print(f'Критическая ошибка конвертации: {e}')
            print("Проверьте установку poppler")
            return[]


if __name__ == "__main__":
    Config.validate()
    converted = PDFConverter.convert("test.pdf")
    print(f'result: {converted}')


