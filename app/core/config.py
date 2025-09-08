from typing import List, Optional
from pydantic import field_validator
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Simplified configuration focused on LangChain agents and vector storage"""
    
    # App Configuration
    APP_NAME: str = "AI Resume Matcher"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    
    # OpenAI Configuration for LangChain
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_MAX_TOKENS: int = 2000
    
    # ChromaDB Vector Store
    CHROMADB_PERSIST_DIRECTORY: str = "./data/vectordb"
    CHROMADB_COLLECTION_NAME: str = "resume_embeddings"
    
    # Embedding Model
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda" for GPU
    
    # File Processing
    MAX_FILE_SIZE_MB: int = 10
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "docx", "txt"]
    TEMP_DIR: str = "./data/temp"
    
    # Data Storage (file-based, no database)
    DATA_DIR: str = "./data"
    RESUMES_DIR: str = "./data/resumes"
    JOBS_DIR: str = "./data/jobs"
    RESULTS_DIR: str = "./data/results"
    
    @field_validator("ALLOWED_FILE_TYPES", mode="before")
    def parse_allowed_file_types(cls, v):
        if isinstance(v, str):
            return [x.strip() for x in v.split(",")]
        return v
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.DATA_DIR,
            self.RESUMES_DIR,
            self.JOBS_DIR,
            self.RESULTS_DIR,
            self.TEMP_DIR,
            self.CHROMADB_PERSIST_DIRECTORY
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore"
    )


# Global settings instance
settings = Settings()

# Ensure directories exist
settings.ensure_directories()


# Create settings instance
settings = Settings()

# Ensure data directories exist
Path(settings.CHROMADB_PERSIST_DIRECTORY).mkdir(parents=True, exist_ok=True)
Path("./data/resumes").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
