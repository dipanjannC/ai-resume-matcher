"""
Utility functions for file operations.
Simple, clean file handling without database dependencies.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, BinaryIO
import pypdf
import docx
from app.core.exceptions import FileParsingException
from app.core.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from various file formats.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Extracted text content
        
    Raises:
        FileParsingException: If file cannot be parsed
    """
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        raise FileParsingException(f"File not found: {file_path}")
    
    try:
        file_extension = file_path_obj.suffix.lower()
        
        if file_extension == '.pdf':
            return _extract_pdf_text(file_path_obj)
        elif file_extension == '.docx':
            return _extract_docx_text(file_path_obj)
        elif file_extension == '.txt':
            return _extract_txt_text(file_path_obj)
        else:
            raise FileParsingException(f"Unsupported file type: {file_extension}")
            
    except FileParsingException:
        raise
    except Exception as e:
        raise FileParsingException(f"Error extracting text from {file_path}: {str(e)}")


def _extract_pdf_text(file_path: Path) -> str:
    """Extract text from PDF file"""
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num}: {str(e)}")
                    continue
        
        if not text.strip():
            raise FileParsingException("No readable text found in PDF")
            
        return text.strip()
        
    except Exception as e:
        raise FileParsingException(f"Failed to parse PDF: {str(e)}")


def _extract_docx_text(file_path: Path) -> str:
    """Extract text from DOCX file"""
    try:
        doc = docx.Document(str(file_path))
        text = ""
        
        # Extract text from paragraphs
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += paragraph.text + "\n"
        
        # Extract text from tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    if cell.text.strip():
                        text += cell.text + " "
                text += "\n"
        
        if not text.strip():
            raise FileParsingException("No readable text found in DOCX")
            
        return text.strip()
        
    except Exception as e:
        raise FileParsingException(f"Failed to parse DOCX: {str(e)}")


def _extract_txt_text(file_path: Path) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        if not text.strip():
            raise FileParsingException("Text file is empty")
            
        return text.strip()
        
    except UnicodeDecodeError:
        # Try with different encoding
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
            return text.strip()
        except Exception as e:
            raise FileParsingException(f"Failed to read text file with encoding: {str(e)}")
    except Exception as e:
        raise FileParsingException(f"Failed to parse text file: {str(e)}")


def save_uploaded_file(file_content: bytes, filename: str) -> str:
    """
    Save uploaded file content to temporary file.
    
    Args:
        file_content: Raw file bytes
        filename: Original filename
        
    Returns:
        Path to saved temporary file
    """
    try:
        file_extension = Path(filename).suffix.lower()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        logger.info(f"Saved uploaded file to temporary location: {temp_file_path}")
        return temp_file_path
        
    except Exception as e:
        raise FileParsingException(f"Failed to save uploaded file: {str(e)}")


def cleanup_temp_file(file_path: str) -> None:
    """Remove temporary file safely"""
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            logger.debug(f"Cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Could not clean up temporary file {file_path}: {str(e)}")


def validate_file_type(filename: str, allowed_types: list) -> bool:
    """
    Validate if file type is allowed.
    
    Args:
        filename: Name of the file
        allowed_types: List of allowed file extensions (without dots)
        
    Returns:
        True if file type is allowed
    """
    if not filename:
        return False
    
    file_extension = Path(filename).suffix.lower().lstrip('.')
    return file_extension in allowed_types


def validate_file_size(file_content: bytes, max_size_mb: int) -> bool:
    """
    Validate file size.
    
    Args:
        file_content: File content in bytes
        max_size_mb: Maximum size in MB
        
    Returns:
        True if file size is within limit
    """
    size_mb = len(file_content) / (1024 * 1024)
    return size_mb <= max_size_mb


if __name__ == "__main__":
    # Example usage
    test_file_path = "example.pdf"
    try:
        text = extract_text_from_file(test_file_path)
        print(f"Extracted Text:\n{text}")
    except FileParsingException as e:
        print(f"Error: {str(e)}")