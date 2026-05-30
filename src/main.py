# src/main.py
#!/usr/bin/env python3
"""
Точка входа в приложение.
Запускает пайплайн обработки документа.
"""

import sys
import hashlib
import cv2
from pathlib import Path

# Добавляем src в путь, чтобы работали импорты
sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DIR
from converter import PDFConverter
from preprocessor import DocumentPreprocessor
from api import YandexClient
from db_manager import DatabaseManager


def process_document(pdf_path: str) -> int | None:
    """Полный пайплайн обработки документа"""
    
    # 1. Инициализация
    ai_client = YandexClient()
    db = DatabaseManager()
    preprocessor = DocumentPreprocessor(output_dir=PROCESSED_DIR)
    
    # 2. Конвертация PDF → PNG
    print(f"Конвертация: {pdf_path}")
    raw_images = PDFConverter.convert(pdf_path)
    if not raw_images:
        print("Не удалось конвертировать PDF")
        return None
    
    # 3. Предобработка
    print("Улучшение изображений для ИИ...")
    enhanced_images = preprocessor.preprocess_batch(raw_images)
    
    # 4. Извлечение метаданных (ТОЛЬКО из шапки первой страницы)
    print("Распознавание шапки...")
    header_image_path = preprocessor.extract_header(enhanced_images[0])
    raw_header_text = ai_client.extract_header_raw_text(header_image_path)
    
    # СТРОГИЙ ДЕБАГ: Смотрим, что вернул API Яндекса
    print(f"[DEBUG] Сырой текст шапки от OCR: '{raw_header_text}'")

    if raw_header_text:
        metadata = ai_client.extract_metadata(raw_header_text)
    else:
        metadata = None
        print("[DEBUG] Передача в YandexGPT отменена, так как сырой текст шапки пуст.")
    
    if metadata:
        print(f"Найдено: {metadata.get('doc_type')} №{metadata.get('doc_number')} от {metadata.get('doc_date')}")
    else:
        print("Метаданные не извлечены или вернулся пустой JSON.")
    
    # 5. Извлечение тела документа
    print("Распознавание тела документа...")
    body_parts = []
    
    # Используем enumerate для точного определения первой страницы (индекс 0)
    for i, img_path in enumerate(enhanced_images):
        if i == 0:  # СТРОГАЯ ЛОГИКА: Это первая страница
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Ошибка чтения изображения OpenCV: {img_path}")
                continue
                
            image_height = img.shape[0]
            crop_threshold = int(image_height * 0.25) 
            text = ai_client.extract_body_text(img_path, crop_y_threshold=crop_threshold)
        else:       # Это вторая и последующие страницы (не режем)
            text = ai_client.extract_body_text(img_path)
            
        if text:
            body_parts.append(text)
            
    body_text = "\n\n".join(body_parts)
    if body_text:
        preview = body_text[:200].replace('\n', ' ')
        print(f"Текст: {preview}...")
    
    # 6. Сохранение в БД
    file_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    doc_id = db.save_document(Path(pdf_path).name, file_hash)
    
    if metadata:
        db.save_metadata(doc_id, metadata, str(metadata))
    if body_text:
        db.save_content(doc_id, body_text, page_count=len(enhanced_images))
    print(f"💾 Сохранено в БД с ID={doc_id}")
    
    # 7. Очистка
    files_to_remove = enhanced_images + [header_image_path]
    DocumentPreprocessor.cleanup(files_to_remove)
    db.close()
    return doc_id


if __name__ == "__main__":
    # Простой CLI: python main.py <путь_к_папке_или_файлу>
    if len(sys.argv) < 2:
        print("Использование: python main.py <путь_к_папке_или_pdf_файлу>")
        sys.exit(1)
    
    input_target = Path(sys.argv[1])
    
    # Проверка существования пути
    if not input_target.exists():
        print(f"❌ Путь не найден: {input_target}")
        sys.exit(1)
        
    pdf_files = []
    
    # Сценарий 1: Передан конкретный файл
    if input_target.is_file():
        if input_target.suffix.lower() == '.pdf':
            pdf_files.append(input_target)
        else:
            print(f"❌ Ошибка: Файл {input_target.name} не является PDF.")
            sys.exit(1)
            
    # Сценарий 2: Передана папка
    elif input_target.is_dir():
        pdf_files = list(input_target.glob("*.pdf")) + list(input_target.glob("*.PDF"))
        if not pdf_files:
            print(f"⚠️ В папке {input_target} не найдено PDF-файлов.")
            sys.exit(0)
            
    print(f"📂 Найдено PDF-файлов для обработки: {len(pdf_files)}")
    print("-" * 40)
    
    # Итеративный запуск конвейера
    success_count = 0
    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"\n🔄 [Файл {i}/{len(pdf_files)}] Запуск пайплайна для: {pdf_path.name}")
        
        # Передаем абсолютный путь в строковом формате
        doc_id = process_document(str(pdf_path.resolve()))
        
        if doc_id:
            success_count += 1
            print(f"✅ Успешно завершено. ID в базе: {doc_id}")
        else:
            print(f"❌ Ошибка обработки файла: {pdf_path.name}")
            
    print("-" * 40)
    print(f"🏁 Обработка завершена. Успешно: {success_count}/{len(pdf_files)}")