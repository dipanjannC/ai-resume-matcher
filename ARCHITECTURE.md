# AI Resume Matcher - Architecture Documentation

## Overview

The AI Resume Matcher is a sophisticated system that leverages LangChain agents and OpenAI's language models to extract structured information from unstructured resume data and match candidates with job requirements. The system has been simplified from a complex database-driven architecture to a streamlined file-based approach focused on LangChain capabilities.

## Architecture Principles

- **Simplicity First**: Removed PostgreSQL/Redis dependencies for easier deployment and maintenance
- **LangChain Focused**: Built around LangChain agents for structured data extraction and processing
- **File-Based Storage**: Uses JSON files and ChromaDB for persistence, eliminating complex database setup
- **Modular Design**: Clean separation of concerns with dedicated service modules
- **Type Safety**: Comprehensive Pydantic models for data validation and structure

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
├─────────────────────────────────────────────────────────────────┤
│                          app/main.py                            │
│                    (REST API Endpoints)                         │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Resume Processing Layer                       │
├─────────────────────────────────────────────────────────────────┤
│                app/services/resume_processor.py                 │
│              (Main orchestrator & coordinator)                  │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  LangChain      │ │   Embeddings    │ │  Vector Store   │
│    Agents       │ │    Service      │ │    Service      │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ langchain_      │ │  embeddings.py  │ │ vector_store.py │
│ agents.py       │ │                 │ │                 │
│                 │ │ sentence-       │ │   ChromaDB      │
│ - Resume Parser │ │ transformers    │ │   Persistent    │
│ - Job Analyzer  │ │                 │ │                 │
│ - Matcher       │ │ Text→Vectors    │ │ Vector Storage  │
│ - Summarizer    │ │                 │ │ & Search        │
└─────────────────┘ └─────────────────┘ └─────────────────┘
        │                                       │
        ▼                                       │
┌─────────────────┐                             │
│ Prompt Manager  │                             │
├─────────────────┤                             │
│ prompt_manager  │                             │
│     .py         │                             │
│                 │                             │
│ - Resume Prompts│                             │
│ - Job Prompts   │                             │
│ - Match Prompts │                             │
│ - Summary Prompts│                            │
└─────────────────┘                             │
        │                                       │
        ▼                                       ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│  Data Models    │ │   File Utils    │ │   Core Utils    │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ models/         │ │ file_utils.py   │ │   config.py     │
│                 │ │                 │ │ exceptions.py   │
│ langchain_      │ │ PDF/DOCX        │ │  logging.py     │
│ models.py       │ │ Processing      │ │                 │
│                 │ │                 │ │ Settings &      │
│ resume_data.py  │ │ Text Extraction │ │ Error Handling  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│   File Utils    │ │  Prompt Manager │ │   Core Utils    │
├─────────────────┤ ├─────────────────┤ ├─────────────────┤
│ file_utils.py   │ │ prompt_manager  │ │   config.py     │
│                 │ │     .py         │ │ exceptions.py   │
│ PDF/DOCX        │ │                 │ │  logging.py     │
│ Processing      │ │ System Prompts  │ │                 │
│                 │ │ Role Prompts    │ │ Settings &      │
│ Text Extraction │ │ User Prompts    │ │ Error Handling  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Core Components

### 1. LangChain Agents (`app/services/langchain_agents.py`)

**Purpose**: Central AI intelligence using OpenAI GPT-3.5-turbo for structured data extraction

**Key Components**:

- **ResumeParsingAgent**: Extracts structured data from unstructured resume text
- **JobAnalysisAgent**: Analyzes job descriptions and extracts requirements
- **MatchingAgent**: Performs intelligent candidate-job matching
- **SummaryEnhancementAgent**: Generates enhanced professional summaries

**Architecture Improvements** (Post-Refactoring):

- **Separated Concerns**: Prompts moved to centralized `prompt_manager`
- **Modular Models**: Pydantic models moved to `app/models/langchain_models.py`
- **Clean Business Logic**: Focused purely on AI orchestration and data processing
- **External Dependencies**: Uses `prompt_manager.get_*_prompt()` for all templates

### 2. Data Models (`app/models/`)

**Purpose**: Centralized data structure definitions for type safety and consistency

**Location**: `app/models/langchain_models.py`

**Pydantic Models for LangChain Output Parsing**:

```python
ResumeParsingOutput:
  - profile: Dict[str, str] (name, title, email, phone, linkedin, location)
  - experience: Dict[str, Any] (total_years, roles, companies, responsibilities, achievements)
  - skills: Dict[str, List[str]] (technical, soft, certifications, languages)
  - topics: Dict[str, List[str]] (domains, specializations, interests)
  - tools_libraries: Dict[str, List[str]] (programming_languages, frameworks, tools, databases)
  - summary: str
  - key_strengths: List[str]

JobParsingOutput:
  - title: str
  - company: str
  - required_skills: List[str]
  - preferred_skills: List[str]
  - experience_years: int
  - education_level: str
  - responsibilities: List[str]
  - requirements: List[str]
  - company_info: str
  - summary: str

MatchAnalysisOutput:
  - skills_match_score: float (0-1)
  - experience_match_score: float (0-1)
  - overall_score: float (0-1)
  - matching_skills: List[str]
  - missing_skills: List[str]
  - strength_areas: List[str]
  - improvement_areas: List[str]
  - match_summary: str
  - recommendation: str
```

**Location**: `app/models/resume_data.py`

**Domain Models for Application Data**:

```python
ResumeData, JobDescription, MatchResult
ProfileInfo, ExperienceInfo, SkillsInfo, TopicsInfo, ToolsLibrariesInfo
```

### 3. Prompt Management (`app/services/prompt_manager.py`)

**Purpose**: Centralized prompt template management for all LangChain operations

**Key Features**:

- **LangChainPromptManager**: Main class for prompt orchestration
- **Modular Prompt Organization**: Separate methods for different AI tasks
- **Template Consistency**: Standardized prompt structure across all agents
- **Easy Maintenance**: Single location for prompt updates and improvements

**Available Prompt Templates**:

```python
# Core LangChain Agent Prompts
get_resume_parsing_prompt()     # Resume → Structured Data
get_job_parsing_prompt()        # Job Description → Requirements
get_matching_prompt()           # Candidate + Job → Match Analysis
get_summary_prompt()            # Resume Data → Enhanced Summary

# Legacy Support (Backward Compatibility)
get_chat_template()             # Traditional chat-based matching
get_full_template()             # Complete template with all components
```

**Prompt Categories**:

- **System Prompts**: Define AI agent roles and capabilities
- **Resume Parsing Prompts**: Extract structured data from unstructured resumes
- **Job Analysis Prompts**: Analyze job descriptions and extract requirements
- **Matching Prompts**: Compare candidates with job requirements
- **Summary Enhancement Prompts**: Generate compelling professional summaries

### 4. Resume Processor (`app/services/resume_processor.py`)

**Purpose**: Main orchestrator coordinating all services for end-to-end processing

**Key Functions**:

- `process_resume_file()`: Complete resume processing pipeline
- `find_best_matches()`: Multi-criteria candidate matching
- `get_candidate_details()`: Retrieve processed candidate information
- `batch_process_resumes()`: Bulk resume processing

**Features**:

- Async processing for better performance
- File-based JSON persistence
- Vector storage integration
- Error handling and logging

### 5. Vector Store (`app/services/vector_store.py`)

**Purpose**: ChromaDB-based vector storage for semantic similarity search

**Key Features**:
- **Persistent Storage**: File-based ChromaDB persistence
- **Semantic Search**: Vector similarity search for resumes
- **Metadata Management**: Rich metadata storage and filtering
- **Collection Management**: Create, reset, and manage collections

**Operations**:
- `add_resume()`: Store resume embeddings
- `search_similar()`: Find similar resumes by vector similarity
- `get_all_candidates()`: Retrieve all stored candidates
- `resume_exists()`: Check if resume already processed

### 4. Embeddings Service (`app/services/embeddings.py`)

**Purpose**: Text embedding generation using sentence-transformers

**Model**: `all-MiniLM-L6-v2` (lightweight, fast, good performance)

**Features**:
- Batch processing support
- Device optimization (CPU/GPU)
- Consistent vector dimensions
- Caching capabilities

### 5. Prompt Manager (`app/services/prompt_manager.py`)

**Purpose**: Centralized management of LangChain prompts

**Prompt Types**:
- **System Prompts**: Define AI agent roles and capabilities
- **Resume Parsing Prompts**: Extract structured data from resumes
- **Job Analysis Prompts**: Analyze job descriptions
- **Matching Prompts**: Compare candidates with jobs

**Features**:
- Template-based prompt management
- Role-based prompt organization
- Easy prompt customization

### 6. File Utilities (`app/utils/file_utils.py`)

**Purpose**: Document processing and text extraction

**Supported Formats**:
- PDF files (using pypdf)
- Word documents (using python-docx)
- Plain text files

**Features**:
- Robust text extraction
- Error handling for corrupted files
- Encoding detection and handling

## Data Flow

### Resume Processing Pipeline

```mermaid
1. File Upload → FastAPI Endpoint
2. Document Processing → file_utils.py
3. Text Extraction → PDF/DOCX/TXT
4. Prompt Retrieval → prompt_manager.py (get_resume_parsing_prompt)
5. LangChain Processing → langchain_agents.py
6. Structured Extraction → langchain_models.py (ResumeParsingOutput)
7. Data Conversion → resume_data.py (ResumeData)
8. Embedding Generation → embeddings.py
9. Vector Storage → vector_store.py (ChromaDB)
10. JSON Persistence → data/processed/
11. Response → Structured Resume Data
```

### Job Matching Pipeline

```mermaid
1. Job Description Input → FastAPI Endpoint
2. Job Prompt Retrieval → prompt_manager.py (get_job_parsing_prompt)
3. Job Analysis → langchain_agents.py
4. Job Requirements Extraction → langchain_models.py (JobParsingOutput)
5. Query Embedding → embeddings.py
6. Vector Search → vector_store.py
7. Candidate Retrieval → Ranked Results
8. Match Prompt Retrieval → prompt_manager.py (get_matching_prompt)
9. Match Analysis → langchain_agents.py
10. Score Calculation → langchain_models.py (MatchAnalysisOutput)
11. Response → Ranked Candidates with Scores
```

### Prompt Management Flow

```mermaid
1. Agent Initialization → langchain_agents.py
2. Prompt Request → prompt_manager.py
3. Template Retrieval → ChatPromptTemplate objects
4. Prompt Injection → LangChain pipeline
5. AI Processing → OpenAI GPT-3.5-turbo
6. Structured Output → Pydantic model validation
7. Data Transformation → Domain models (resume_data.py)
```

## Storage Architecture

### File System Structure

```
data/
├── processed/           # Processed resume JSON files
│   ├── candidate_001.json
│   ├── candidate_002.json
│   └── ...
├── vectordb/           # ChromaDB persistent storage
│   ├── chroma.sqlite3
│   └── collections/
└── resumes/            # Original resume files
    ├── sample_resume_john.txt
    └── sample_resume_sarah.txt

app/models/             # Centralized data models
├── __init__.py
├── langchain_models.py # Pydantic models for LangChain
└── resume_data.py      # Domain models for application
```

### Data Persistence

1. **Structured Data**: JSON files in `data/processed/`
2. **Vector Data**: ChromaDB in `data/vectordb/`
3. **Raw Files**: Original resumes in `data/resumes/`
4. **Code Models**: Pydantic/dataclass models in `app/models/`

### Model Organization

- **LangChain Models** (`app/models/langchain_models.py`): Output parsing schemas for AI agents
- **Domain Models** (`app/models/resume_data.py`): Business logic data structures
- **Type Safety**: Centralized model definitions ensure consistency across services

## Configuration Management

### Settings (`app/core/config.py`)

```python
# OpenAI Configuration
OPENAI_API_KEY: str
OPENAI_MODEL: str = "gpt-3.5-turbo"

# ChromaDB Configuration
CHROMADB_PERSIST_DIRECTORY: str = "./data/vectordb"

# File Processing
MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
SUPPORTED_FORMATS: List[str] = [".pdf", ".docx", ".txt"]

# Embedding Configuration
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DEVICE: str = "cpu"
```

## API Endpoints

### Core Endpoints

1. **POST /upload-resume**
   - Upload and process resume files
   - Returns structured resume data

2. **POST /find-matches**
   - Find candidates matching job description
   - Returns ranked candidate list with scores

3. **GET /candidates**
   - List all processed candidates
   - Supports filtering and pagination

4. **GET /candidate/{id}**
   - Get detailed candidate information
   - Includes full structured data

## Performance Considerations

### Optimizations

1. **Async Processing**: FastAPI async endpoints for better concurrency
2. **Batch Operations**: Bulk resume processing capabilities
3. **Vector Caching**: ChromaDB persistent storage reduces reprocessing
4. **Lightweight Embeddings**: MiniLM model for fast processing
5. **Structured Logging**: Comprehensive logging for monitoring

### Scalability

- **Horizontal Scaling**: Stateless design supports multiple instances
- **Storage Scaling**: File-based storage easily movable to cloud storage
- **Processing Scaling**: Can add dedicated worker processes for heavy operations

## Error Handling

### Exception Management

1. **Custom Exceptions**: `VectorStoreException`, domain-specific errors
2. **Graceful Degradation**: Fallback behaviors for failed operations
3. **Comprehensive Logging**: Structured logging for debugging
4. **Input Validation**: Pydantic models ensure data integrity

## Security Considerations

1. **API Key Management**: Environment-based configuration
2. **File Validation**: Size and format restrictions
3. **Input Sanitization**: Validated through Pydantic models
4. **Error Information**: Careful error message sanitization

## Development Setup

### Prerequisites

```bash
# Python 3.8+
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Environment configuration
cp .env.example .env
# Add your OPENAI_API_KEY
```

### Running the Application

```bash
# Development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Architecture Improvements & Refactoring Benefits

### Modular Design Implementation

The system has been refactored to implement a modular architecture with clear separation of concerns:

#### 1. **Centralized Prompt Management**

- **Before**: Prompts scattered throughout `langchain_agents.py`
- **After**: Centralized in `app/services/prompt_manager.py`
- **Benefits**:
  - Single source of truth for all AI prompts
  - Easy prompt tuning and A/B testing
  - Template reusability across different agents
  - Consistent prompt formatting and structure

#### 2. **Separated Data Models**

- **Before**: Pydantic models mixed with business logic
- **After**: Dedicated `app/models/langchain_models.py` for AI output parsing
- **Benefits**:
  - Type safety and validation consistency
  - Clear contract between AI agents and application
  - Easier model evolution and versioning
  - Reusable model definitions

#### 3. **Clean Business Logic**

- **Before**: Monolithic `langchain_agents.py` with mixed responsibilities
- **After**: Focused agents with external dependencies for prompts and models
- **Benefits**:
  - Reduced complexity and improved readability
  - Better testability with mocked dependencies
  - Easier maintenance and debugging
  - Clear single responsibility principle adherence

### Code Quality Improvements

1. **Import Management**: Proper module imports with clear dependency structure
2. **Error Handling**: Consistent error propagation and logging
3. **Documentation**: Self-documenting code with clear model definitions
4. **Maintainability**: Modular structure supports easier refactoring and updates

### Development Workflow Benefits

- **Prompt Engineering**: Centralized prompts enable rapid iteration
- **Model Updates**: Isolated model changes don't affect business logic
- **Testing**: Mocking external dependencies is straightforward
- **Debugging**: Clear separation makes issue isolation easier

## Future Enhancements

### Planned Improvements

1. **Advanced Matching**: ML-based scoring algorithms
2. **Multi-language Support**: International resume processing
3. **Real-time Processing**: WebSocket-based live updates
4. **Analytics Dashboard**: Candidate and matching analytics
5. **API Authentication**: JWT-based authentication system

### Technology Upgrades

1. **Vector Database**: Transition to production vector DB (Pinecone, Weaviate)
2. **LLM Options**: Support for additional language models
3. **Cloud Deployment**: Container-based cloud deployment
4. **Monitoring**: APM and performance monitoring integration

## Conclusion

The AI Resume Matcher represents a modern, simplified approach to AI-powered recruitment technology. By focusing on LangChain agents and eliminating complex database dependencies, the system provides powerful resume processing capabilities while maintaining simplicity and ease of deployment.

The architecture supports both rapid development and production deployment, with clear separation of concerns and comprehensive error handling. The file-based approach makes it easy to understand, debug, and extend while providing the foundation for future scalability improvements.
