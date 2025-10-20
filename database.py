import sqlite3
import os
import logging
import json

logger = logging.getLogger(__name__)

# Define absolute database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regulations.db")

def migrate_db():
    """Database migration to add missing columns"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Get existing columns
        c.execute("PRAGMA table_info(regulations)")
        existing = {row[1] for row in c.fetchall()}  # column names
        # Columns expected
        required = {
            "id": "TEXT PRIMARY KEY",
            "query": "TEXT",
            "source_type": "TEXT",
            "confirmed_source": "TEXT",
            "language": "TEXT",
            "final_summary": "TEXT",
            "refs_json": "TEXT"
        }
        # Add any missing columns
        for col, coltype in required.items():
            if col not in existing:
                c.execute(f"ALTER TABLE regulations ADD COLUMN {col} {coltype}")
        conn.commit()
    except Exception as e:
        logger.error(f"Error migrating database: {str(e)}")
    finally:
        conn.close()

def init_db():
    """Initialize SQLite database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS regulations (
            id TEXT PRIMARY KEY,
            query TEXT,
            source_type TEXT,
            confirmed_source TEXT,
            language TEXT,
            final_summary TEXT,
            refs_json TEXT
        )''')
        
        # Add indexes for faster queries on various reports
        c.execute('''CREATE INDEX IF NOT EXISTS idx_query ON regulations(query)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_source_type ON regulations(source_type)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_language ON regulations(language)''')
        c.execute('''CREATE INDEX IF NOT EXISTS idx_confirmed_source ON regulations(confirmed_source)''')
        
        conn.commit()
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
    finally:
        conn.close()
    migrate_db()  # Ensure schema is up-to-date

def save_report(report_id, query, source_type, confirmed_source, language, final_summary, references):
    """Save a report to the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT INTO regulations (id, query, source_type, confirmed_source, language, final_summary, refs_json)
                     VALUES (?, ?, ?, ?, ?, ?, ?)''',
                  (report_id, query, source_type, confirmed_source, language, final_summary,
                   json.dumps(references, ensure_ascii=False)))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return False

def delete_report(report_id):
    """Delete a report from the database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM regulations WHERE id = ?', (report_id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        return False