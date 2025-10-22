import sqlite3
import os
import logging
import json
import threading
import time
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Define absolute database path
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regulations.db")

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    url: str = f"sqlite:///{DB_PATH}"
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False
    timeout: int = 30
    check_same_thread: bool = False


class DatabaseConnectionPool:
    """Simple connection pool for SQLite database."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pool = []
        self.lock = threading.Lock()
        self.active_connections = 0
        self.max_connections = config.pool_size + config.max_overflow
        
    def get_connection(self) -> sqlite3.Connection:
        """Get a database connection from the pool."""
        with self.lock:
            if self.pool:
                conn = self.pool.pop()
                self.active_connections += 1
                return conn
            elif self.active_connections < self.max_connections:
                conn = self._create_connection()
                self.active_connections += 1
                return conn
            else:
                raise Exception("Connection pool exhausted")
    
    def return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        with self.lock:
            if self.active_connections > 0:
                self.active_connections -= 1
                if len(self.pool) < self.config.pool_size:
                    self.pool.append(conn)
                else:
                    conn.close()
    
    def _create_connection(self) -> sqlite3.Connection:
        """Create a new database connection."""
        conn = sqlite3.connect(
            DB_PATH,
            timeout=self.config.timeout,
            check_same_thread=self.config.check_same_thread
        )
        conn.row_factory = sqlite3.Row
        return conn
    
    def close_all(self) -> None:
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.pool:
                conn.close()
            self.pool.clear()
            self.active_connections = 0


# Global connection pool
_db_pool: Optional[DatabaseConnectionPool] = None


def get_db_pool() -> DatabaseConnectionPool:
    """Get the global database connection pool."""
    global _db_pool
    if _db_pool is None:
        config = DatabaseConfig()
        _db_pool = DatabaseConnectionPool(config)
    return _db_pool


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    pool = get_db_pool()
    conn = None
    try:
        conn = pool.get_connection()
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            pool.return_connection(conn)

def migrate_db():
    """Database migration to add missing columns and indexes"""
    try:
        with get_db_connection() as conn:
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
                "refs_json": "TEXT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
            
            # Add any missing columns
            for col, coltype in required.items():
                if col not in existing:
                    c.execute(f"ALTER TABLE regulations ADD COLUMN {col} {coltype}")
            
            conn.commit()
            logger.info("Database migration completed")
            
    except Exception as e:
        logger.error(f"Error migrating database: {str(e)}")
        raise

def init_db():
    """Initialize SQLite database with optimized schema and indexes"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Create table with optimized schema
            c.execute('''CREATE TABLE IF NOT EXISTS regulations (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                source_type TEXT NOT NULL,
                confirmed_source TEXT NOT NULL,
                language TEXT NOT NULL,
                final_summary TEXT NOT NULL,
                refs_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            # Create optimized indexes for common queries
            indexes = [
                # Single column indexes
                "CREATE INDEX IF NOT EXISTS idx_query ON regulations(query)",
                "CREATE INDEX IF NOT EXISTS idx_source_type ON regulations(source_type)",
                "CREATE INDEX IF NOT EXISTS idx_language ON regulations(language)",
                "CREATE INDEX IF NOT EXISTS idx_created_at ON regulations(created_at DESC)",
                
                # Composite indexes for common query patterns
                "CREATE INDEX IF NOT EXISTS idx_query_source ON regulations(query, source_type)",
                "CREATE INDEX IF NOT EXISTS idx_source_language ON regulations(source_type, language)",
                "CREATE INDEX IF NOT EXISTS idx_language_created ON regulations(language, created_at DESC)",
                
                # Full-text search index (if FTS is available)
                "CREATE INDEX IF NOT EXISTS idx_summary_text ON regulations(final_summary)"
            ]
            
            for index_sql in indexes:
                try:
                    c.execute(index_sql)
                except sqlite3.Error as e:
                    logger.warning(f"Index creation failed: {e}")
            
            # Enable WAL mode for better concurrency
            c.execute("PRAGMA journal_mode=WAL")
            
            # Optimize SQLite settings
            c.execute("PRAGMA synchronous=NORMAL")
            c.execute("PRAGMA cache_size=10000")
            c.execute("PRAGMA temp_store=MEMORY")
            
            conn.commit()
            logger.info("Database initialized with optimized schema")
            
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        migrate_db()  # Ensure schema is up-to-date

def save_report(report_id: str, query: str, source_type: str, confirmed_source: str, 
                language: str, final_summary: str, references: List[Dict[str, Any]]) -> bool:
    """Save a report to the database with optimized connection handling"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Use prepared statement for better performance
            c.execute('''INSERT INTO regulations 
                        (id, query, source_type, confirmed_source, language, final_summary, refs_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
                      (report_id, query, source_type, confirmed_source, language, final_summary,
                       json.dumps(references, ensure_ascii=False)))
            
            conn.commit()
            logger.info(f"Report saved successfully: {report_id}")
            return True
            
    except Exception as e:
        logger.error(f"Error saving report: {str(e)}")
        return False


def delete_report(report_id: str) -> bool:
    """Delete a report from the database with optimized connection handling"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Use prepared statement
            result = c.execute('DELETE FROM regulations WHERE id = ?', (report_id,))
            rows_affected = result.rowcount
            
            conn.commit()
            
            if rows_affected > 0:
                logger.info(f"Report deleted successfully: {report_id}")
                return True
            else:
                logger.warning(f"Report not found: {report_id}")
                return False
                
    except Exception as e:
        logger.error(f"Error deleting report: {str(e)}")
        return False


def get_report(report_id: str) -> Optional[Dict[str, Any]]:
    """Get a report by ID"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''SELECT * FROM regulations WHERE id = ?''', (report_id,))
            row = c.fetchone()
            
            if row:
                return dict(row)
            return None
            
    except Exception as e:
        logger.error(f"Error getting report: {str(e)}")
        return None


def get_reports_by_query(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get reports by query with pagination"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''SELECT * FROM regulations 
                        WHERE query LIKE ? 
                        ORDER BY created_at DESC 
                        LIMIT ?''', (f"%{query}%", limit))
            rows = c.fetchall()
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Error getting reports by query: {str(e)}")
        return []


def get_reports_by_language(language: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get reports by language with pagination"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''SELECT * FROM regulations 
                        WHERE language = ? 
                        ORDER BY created_at DESC 
                        LIMIT ?''', (language, limit))
            rows = c.fetchall()
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Error getting reports by language: {str(e)}")
        return []


def get_recent_reports(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent reports"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''SELECT * FROM regulations 
                        ORDER BY created_at DESC 
                        LIMIT ?''', (limit,))
            rows = c.fetchall()
            
            return [dict(row) for row in rows]
            
    except Exception as e:
        logger.error(f"Error getting recent reports: {str(e)}")
        return []


def get_database_stats() -> Dict[str, Any]:
    """Get database statistics"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            
            # Get table info
            c.execute("SELECT COUNT(*) FROM regulations")
            total_reports = c.fetchone()[0]
            
            # Get language distribution
            c.execute("SELECT language, COUNT(*) FROM regulations GROUP BY language")
            language_dist = dict(c.fetchall())
            
            # Get source type distribution
            c.execute("SELECT source_type, COUNT(*) FROM regulations GROUP BY source_type")
            source_dist = dict(c.fetchall())
            
            # Get recent activity
            c.execute("SELECT COUNT(*) FROM regulations WHERE created_at > datetime('now', '-7 days')")
            recent_reports = c.fetchone()[0]
            
            return {
                "total_reports": total_reports,
                "language_distribution": language_dist,
                "source_type_distribution": source_dist,
                "recent_reports_7_days": recent_reports,
                "database_path": DB_PATH
            }
            
    except Exception as e:
        logger.error(f"Error getting database stats: {str(e)}")
        return {}


def cleanup_old_reports(days: int = 30) -> int:
    """Clean up reports older than specified days"""
    try:
        with get_db_connection() as conn:
            c = conn.cursor()
            c.execute('''DELETE FROM regulations 
                        WHERE created_at < datetime('now', '-{} days')'''.format(days))
            rows_deleted = c.rowcount
            conn.commit()
            
            logger.info(f"Cleaned up {rows_deleted} old reports")
            return rows_deleted
            
    except Exception as e:
        logger.error(f"Error cleaning up old reports: {str(e)}")
        return 0