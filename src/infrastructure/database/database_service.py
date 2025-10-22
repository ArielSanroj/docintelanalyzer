"""
Database service with connection pooling and optimization.
Provides a clean interface for database operations.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError

# Local imports
from ...core.exceptions import DatabaseError, DatabaseConnectionError, DatabaseQueryError
from ...core.logging_config import get_logger, log_execution_time

logger = get_logger(__name__)


class DatabaseService:
    """
    Database service with connection pooling and optimization.
    """
    
    def __init__(
        self,
        url: str,
        pool_size: int = 10,
        max_overflow: int = 20,
        echo: bool = False
    ):
        """
        Initialize database service.
        
        Args:
            url: Database URL
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            echo: Enable SQL query logging
        """
        self.url = url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.echo = echo
        
        # Initialize engine and session factory
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
        
        # Statistics
        self.stats = {
            "connections_created": 0,
            "connections_closed": 0,
            "queries_executed": 0,
            "errors": 0
        }
    
    def initialize(self) -> None:
        """Initialize database connection."""
        try:
            self.engine = create_engine(
                self.url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                echo=self.echo,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            self.SessionLocal = sessionmaker(
                bind=self.engine,
                autocommit=False,
                autoflush=False
            )
            
            self._initialized = True
            logger.info(f"Database service initialized: {self.url}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            raise DatabaseConnectionError(f"Database initialization failed: {e}")
    
    @contextmanager
    def get_session(self):
        """Get database session with automatic cleanup."""
        if not self._initialized:
            self.initialize()
        
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.stats["errors"] += 1
            logger.error(f"Database session error: {e}")
            raise DatabaseQueryError(f"Database operation failed: {e}")
        finally:
            session.close()
    
    @log_execution_time
    def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL query.
        
        Args:
            query: SQL query string
            parameters: Query parameters
        
        Returns:
            Query results as list of dictionaries
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), parameters or {})
                rows = result.fetchall()
                
                # Convert to list of dictionaries
                columns = result.keys()
                results = [dict(zip(columns, row)) for row in rows]
                
                self.stats["queries_executed"] += 1
                logger.debug(f"Query executed successfully: {len(results)} rows")
                
                return results
                
        except SQLAlchemyError as e:
            self.stats["errors"] += 1
            logger.error(f"Query execution failed: {e}")
            raise DatabaseQueryError(f"Query execution failed: {e}")
    
    @log_execution_time
    def execute_update(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Execute an update/insert/delete query.
        
        Args:
            query: SQL query string
            parameters: Query parameters
        
        Returns:
            Number of affected rows
        """
        try:
            with self.get_session() as session:
                result = session.execute(text(query), parameters or {})
                affected_rows = result.rowcount
                
                self.stats["queries_executed"] += 1
                logger.debug(f"Update executed successfully: {affected_rows} rows affected")
                
                return affected_rows
                
        except SQLAlchemyError as e:
            self.stats["errors"] += 1
            logger.error(f"Update execution failed: {e}")
            raise DatabaseQueryError(f"Update execution failed: {e}")
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Get database connection information."""
        if not self._initialized:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        return {
            "url": self.url,
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "stats": self.stats
        }
    
    def health_check(self) -> bool:
        """Check database health."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def close(self) -> None:
        """Close database connections."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connections closed")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.close()

