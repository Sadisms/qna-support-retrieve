from typing import Any, Dict, Optional

from fastapi import HTTPException


class QnAException(Exception):
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DatabaseException(QnAException):
    pass


class EmbeddingException(QnAException):
    pass


class LLMException(QnAException):
    pass


class VectorStoreException(QnAException):
    pass


def create_http_exception(status_code: int, message: str, details: Optional[Dict[str, Any]] = None) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "message": message,
            "details": details or {}
        }
    )
