# src/db_manager.py
"""
Модуль для работы с базой данных (SQLite).
"""

import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

class DatabaseManager:
    """Управление хранением документов и их метаданных."""
    
    def __init__(self, db_path: str = "documents.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON;")  # Включаем FK
        self._init_tables()
    
    def _init_tables(self):
        """Создаёт таблицы, если их нет."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица документов
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'processed'
                )
            """)
            
            # Таблица метаданных
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    doc_id INTEGER PRIMARY KEY,
                    doc_number TEXT,
                    doc_date TEXT,
                    doc_type TEXT,
                    issuer TEXT,
                    raw_json TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            # Таблица полного текста
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_content (
                    doc_id INTEGER PRIMARY KEY,
                    full_text TEXT,
                    page_count INTEGER,
                    extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """)
            
            conn.commit()
    
    def save_document(self, filename: str, file_hash: str) -> int:
        """Сохраняет информацию о файле, возвращает ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR IGNORE INTO documents (original_filename, file_hash) VALUES (?, ?)",
                (filename, file_hash)
            )
            conn.commit()
            cursor.execute("SELECT id FROM documents WHERE file_hash = ?", (file_hash,))
            return cursor.fetchone()[0]
    
    def save_metadata(self, doc_id: int, metadata: Dict[str, Optional[str]], raw_json: str):
        """Сохраняет извлечённые метаданные."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO document_metadata 
                (doc_id, doc_number, doc_date, doc_type, issuer, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                metadata.get('doc_number'),
                metadata.get('doc_date'),
                metadata.get('doc_type'),
                metadata.get('issuer'),
                raw_json
            ))
            conn.commit()
    
    def save_content(self, doc_id: int, full_text: str, page_count: int = 1):
        """Сохраняет распознанный текст документа."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO document_content 
                (doc_id, full_text, page_count)
                VALUES (?, ?, ?)
            """, (doc_id, full_text, page_count))
            conn.commit()
    
    def get_document(self, doc_id: int) -> Optional[Dict]:
        """Получает полную информацию о документе по ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT d.*, m.doc_number, m.doc_date, m.doc_type, m.issuer, c.full_text
                FROM documents d
                LEFT JOIN document_metadata m ON d.id = m.doc_id
                LEFT JOIN document_content c ON d.id = c.doc_id
                WHERE d.id = ?
            """, (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    def close(self):
        if self.conn:
            self.conn.close()