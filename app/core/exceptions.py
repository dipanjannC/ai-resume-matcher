from fastapi import HTTPException, status
from typing import Optional

class ResumeMatcherException(Exception):
    """Base exception for Resume Matcher application"""
    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class FileParsingException(ResumeMatcherException):
    """Raised when file parsing fails"""
    pass


class EmbeddingGenerationException(ResumeMatcherException):
    """Raised when embedding generation fails"""
    pass


class VectorStoreException(ResumeMatcherException):
    """Raised when vector store operations fail"""
    pass


class DatabaseException(ResumeMatcherException):
    """Raised when database operations fail"""
    pass


class FileUploadException(ResumeMatcherException):
    """Raised when file upload fails"""
    pass


# HTTP Exceptions
class HTTPNotFoundException(HTTPException):
    def __init__(self, detail: str = "Resource not found"):
        super().__init__(status_code=status.HTTP_404_NOT_FOUND, detail=detail)


class HTTPBadRequestException(HTTPException):
    def __init__(self, detail: str = "Bad request"):
        super().__init__(status_code=status.HTTP_400_BAD_REQUEST, detail=detail)


class HTTPInternalServerException(HTTPException):
    def __init__(self, detail: str = "Internal server error"):
        super().__init__(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)
