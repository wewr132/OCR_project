# src/main.py
#!/usr/bin/env python3
"""
Точка входа в приложение.
Запускает пайплайн обработки документа.
"""

import sys
import hashlib
from pathlib import Path

# Добавляем src в путь, чтобы работали импорты
sys.path.insert(0, str(Path(__file__).parent))

from config import PROCESSED_DIR
from converter import PDFConverter
from preprocessor import DocumentPreprocessor
from api import YandexVLMClient
from db_manager import DatabaseManager


def process_document(pdf_path: str) -> int | None:
    """Полный пайплайн обработки документа"""
    
    # 1. Инициализация
    ai_client = YandexVLMClient()
    db = DatabaseManager()
    preprocessor = DocumentPreprocessor(output_dir=PROCESSED_DIR)
    
    # 2. Конвертация PDF → PNG (высокое разрешение)
    print(f"📄 Конвертация: {pdf_path}")
    raw_images = PDFConverter.convert(pdf_path)
    if not raw_images:
        print("❌ Не удалось конвертировать PDF")
        return None
    
    # 3. Предобработка для VLM (CLAHE + шумоподавление, БЕЗ бинаризации)
    print("🖼️  Улучшение изображений для ИИ...")
    enhanced_images = preprocessor.preprocess_batch(raw_images)
    
    # 4. Извлечение метаданных (ТОЛЬКО из шапки первой страницы)
    first_page = enhanced_images[0]
    header_path = preprocessor.extract_header(first_page)
    
    print("🤖 Распознавание шапки...")
    metadata = ai_client.extract_metadata(header_path)
    if metadata:
        print(f"✅ Найдено: {metadata.get('doc_type')} №{metadata.get('doc_number')} от {metadata.get('doc_date')}")
    
    # 5. Извлечение тела документа (ВСЕ страницы, без дублирования шапки)
    print("🤖 Распознавание тела документа...")
    body_parts = []
    for img_path in enhanced_images:
        text = ai_client.extract_body_text(img_path)
        if text:
            body_parts.append(text)
    body_text = "\n\n".join(body_parts)
    if body_text:
        preview = body_text[:200].replace('\n', ' ')
        print(f"✅ Текст: {preview}...")
    
    # 6. Сохранение в БД
    file_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    doc_id = db.save_document(Path(pdf_path).name, file_hash)
    
    if metadata:
        db.save_metadata(doc_id, metadata, str(metadata))
    if body_text:
        db.save_content(doc_id, body_text, page_count=len(enhanced_images))
    print(f"💾 Сохранено в БД с ID={doc_id}")
    
    # 7. 🗑️ ОЧИСТКА: удаляем временные файлы только после успешной записи
    files_to_remove = enhanced_images + [header_path]
    DocumentPreprocessor.cleanup(files_to_remove)
    
    return doc_id


if __name__ == "__main__":
    # Простой CLI: python main.py path/to/file.pdf
    if len(sys.argv) < 2:
        print("Использование: python main.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    if not Path(pdf_file).exists():
        print(f"❌ Файл не найден: {pdf_file}")
        sys.exit(1)
    
    process_document(pdf_file)