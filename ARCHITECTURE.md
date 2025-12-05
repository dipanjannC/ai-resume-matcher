# AI Resume Matcher - Architecture

## System Overview

The AI Resume Matcher is a **LangChain-powered resume matching system** built with a modular, file-based architecture that emphasizes simplicity and maintainability.

## Core Architecture

## Core Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │    FastAPI       │    │   CLI Tools     │
│   (Sidebar Nav) │    │   (Optional)     │    │   (Demos)       │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Resume Processor      │
                    │   (Main Orchestrator)   │
                    └────────────┬────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
┌───────▼────────┐    ┌──────────▼─────────┐    ┌─────────▼────────┐
│ LangChain      │    │ Memory Service     │    │ Vector Store     │
│ Agents         │    │ (Mem0 / Graphiti)  │    │ (ChromaDB)       │
│ (Multi-LLM)    │    │                    │    │                  │
│ • Groq         │    │ • User Context     │    │ • Embeddings     │
│ • Gemini       │    │ • Long-term Memory │    │ • Similarity     │
│ • OpenAI       │    │                    │    │   Search         │
└────────────────┘    └────────────────────┘    └──────────────────┘
```

## Key Components

### 1. **Resume Processor** (`app/services/resume_processor.py`)
- **Role**: Main orchestrator for all resume operations
- **Responsibilities**:
  - Coordinates file upload and text extraction
  - Manages LangChain agent interactions
  - Handles vector database storage
  - Manages file-based persistence

### 2. **LangChain Agents** (`app/services/langchain_agents.py`)
- **Role**: AI-powered document processing with Multi-LLM support
- **Features**:
  - **Robust Fallback**: Automatically switches between Groq, Gemini, and OpenAI
  - Structured resume parsing using Pydantic models
  - Job description analysis
  - Intelligent matching and scoring

### 3. **Memory Service** (`app/services/memory_service.py`)
- **Role**: Context management and personalization
- **Providers**:
  - **Mem0**: Local/Cloud memory storage
  - **Graphiti**: Knowledge graph-based memory
- **Features**:
  - Stores user preferences and interaction history
  - Enhances matching with historical context

### 4. **Vector Store** (`app/services/vector_store.py`)
- **Role**: Semantic similarity search
- **Technology**: ChromaDB with persistent storage
- **Features**:
  - Resume embedding storage
  - Metadata-rich search capabilities
  - Cosine similarity matching

### 5. **Data Pipeline** (`app/services/data_pipeline.py`)
- **Role**: Bulk data processing
- **Features**:
  - CSV/JSON batch uploads
  - Sample data loading
  - Progress tracking and error handling

## Data Flow

### Resume Processing Flow
```
1. File Upload (PDF/DOCX/TXT)
   ↓
2. Text Extraction (file_utils.py)
   ↓
3. LangChain Parsing (Multi-LLM Fallback)
   ↓
4. Vector Embedding (sentence-transformers)
   ↓
5. Storage (ChromaDB + JSON files)
```

### Job Matching Flow
```
1. Job Description Input
   ↓
2. LangChain Analysis (with Memory Context)
   ↓
3. Vector Search (semantic similarity)
   ↓
4. AI-Powered Scoring
   ↓
5. Ranked Results with Explanations
```

## Storage Strategy

### File-Based Storage (`./data/`)
```
data/
├── resumes/          # Structured resume data (JSON)
├── jobs/             # Job descriptions (JSON)  
├── vectordb/         # ChromaDB persistence
├── results/          # Matching results
└── temp/             # Temporary file processing
```

### Benefits of File-Based Approach
- **Simplicity**: No database setup required
- **Portability**: Easy to backup and migrate
- **Transparency**: Human-readable data storage
- **Development Speed**: Fast iteration and debugging

## Technology Stack

### Core Technologies
- **Python 3.12+**: Primary language
- **LangChain**: LLM orchestration and structured output
- **LLM Providers**: Groq (Llama 3), Gemini (Pro), OpenAI (GPT-3.5/4)
- **Memory**: Mem0, Graphiti
- **ChromaDB**: Vector database with persistence
- **Sentence Transformers**: CPU-optimized embeddings

### Web Interface
- **Streamlit**: Primary web interface with caching
- **FastAPI**: Optional REST API server
- **Plotly**: Interactive visualizations and charts

### Data Processing
- **Pydantic**: Data validation and structured models
- **PyPDF**: PDF text extraction
- **python-docx**: Word document processing

## Design Patterns

### 1. **Service Layer Pattern**
- Clear separation between UI, business logic, and data
- Each service handles a specific domain (resumes, jobs, matching)
- Async/await for non-blocking operations

### 2. **Repository Pattern**
- Vector store abstracts ChromaDB operations
- File utilities handle different document formats
- Consistent interfaces across storage types

### 3. **Agent Pattern (LangChain)**
- Structured prompts with role definitions
- Pydantic models for consistent output parsing
- Error handling and response cleaning

## Scalability Considerations

### Current Architecture (MVP)
- **Suitable for**: 100-1000 resumes, small teams, demos
- **Limitations**: Single-process, file-based storage

### Future Scaling Options
- **Database Migration**: PostgreSQL + pgvector for production
- **Microservices**: Split services into containers
- **Queue System**: Background job processing
- **Caching Layer**: Redis for frequent operations

## Security & Privacy

### Current Implementation
- **Local Processing**: All data stays on-premises
- **API Keys**: Environment variable configuration
- **File Validation**: Type and size restrictions

### Production Considerations
- **Data Encryption**: Encrypt sensitive resume data
- **Access Control**: User authentication and authorization  
- **Audit Logging**: Track data access and operations
- **GDPR Compliance**: Data retention and deletion policies

## Development Workflow

### Local Development
1. **Environment Setup**: Virtual environment + requirements.txt
2. **Configuration**: `.env` file with API keys
3. **Data Loading**: Sample data via data pipeline
4. **Testing**: Unit tests and integration demos

### Deployment Options
- **Single VM**: Simple deployment with Docker
- **Managed Services**: GCP App Engine, AWS ECS
- **Kubernetes**: For production scalability