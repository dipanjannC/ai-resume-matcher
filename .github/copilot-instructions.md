# AI Resume Matcher - Copilot Instructions

## Architecture Overview

This is a **LangChain-powered resume matching system** using file-based storage (no traditional database). The core pattern:
- **LangChain agents** parse unstructured resumes/jobs into structured data using **Groq by default** (not OpenAI)
- **ChromaDB** provides vector storage with semantic search
- **Streamlit** serves the interactive web interface with caching
- **File-based persistence** in `./data/` directory (JSON + vector embeddings)

## Key Components & Data Flow

### Core Services (`app/services/`)
- `resume_processor.py` - Main orchestrator, coordinates all resume operations
- `langchain_agents.py` - LLM-powered parsing using structured prompts and Pydantic models  
- `vector_store.py` - ChromaDB wrapper with health checks and auto-recovery
- `job_processor.py` - Job description processing and storage
- `data_pipeline.py` - Bulk operations for CSV/JSON uploads

### Data Flow Pattern
```
File Upload → Text Extraction → LangChain Agent → Structured Data → Vector Embedding → ChromaDB + JSON Storage
```

## Critical Development Workflows

### Environment Setup
```bash
# Required API keys (check .env.example)
export GROQ_API_KEY=your_groq_api_key_here  # Primary LLM provider
export OPENAI_API_KEY=your_openai_key       # Fallback (optional)

# Install dependencies
uv pip install -r requirements.txt
```

### Running the Application
```bash
# Streamlit web interface (primary interface) 
python run_streamlit.py

# FastAPI server mode
python app/main.py
```

### Terminal Output Workaround
**IMPORTANT**: Due to Copilot terminal output bugs, always pipe command output:
```bash
command_name > output.txt 2>&1
# Then read output.txt to see results
```

## Project-Specific Patterns

### LLM Service Configuration
**Key**: System uses **Groq by default**, not OpenAI. See `app/services/llm.py`:
```python
# Default model: "gemma2-9b-it" via Groq
self.llm = LLMService().get_groq()  # Line 39 in langchain_agents.py
```

### LangChain Integration Pattern
- **Structured Output**: All agents use Pydantic models (`app/models/langchain_models.py`)
- **Prompt Management**: Centralized in `prompt_manager.py` - modify prompts here
- **JSON Cleaning**: `_clean_json_response()` handles LLM output sanitization

### Data Models & Storage
- **Dataclasses**: `app/models/resume_data.py` defines core structures (ProfileInfo, ExperienceInfo, etc.)
- **File Storage**: JSON files in `./data/resumes/` and `./data/jobs/`
- **Vector Metadata**: ChromaDB stores embeddings + searchable metadata

### Streamlit Caching Strategy
```python
@st.cache_data(ttl=300)  # 5-minute cache
def get_stored_jobs():
    return asyncio.run(job_processor.list_stored_jobs())
```
**Always call `st.cache_data.clear()`** after bulk operations to refresh UI.

### Vector Store Health Checks
ChromaDB has auto-recovery via `_ensure_collection_health()` in `vector_store.py`. If collection becomes inaccessible, it reinitializes automatically.

## Integration Points

### File Processing Pipeline
- **Supported**: PDF, DOCX, TXT (see `config.py` ALLOWED_FILE_TYPES)
- **Text Extraction**: `file_utils.py` handles format-specific extraction
- **Temp Files**: Processed in `./data/temp/` then cleaned up

### Async Operation Pattern
```python
# Standard async service call pattern
try:
    result = await resume_processor.process_resume_file(file_path)
except ResumeMatcherException as e:
    logger.error(f"Processing failed: {e}")
```

### Error Handling Convention
- Custom exceptions in `app/core/exceptions.py`
- Structured logging: `logger = get_logger(__name__)`
- Fallback mechanisms when LangChain agents fail

## Development Patterns

### Adding New LangChain Agents
1. Define Pydantic output model in `langchain_models.py`
2. Add prompt template to `prompt_manager.py`  
3. Implement agent method in `langchain_agents.py` using `_clean_json_response()`
4. Add service method in appropriate processor class

### Streamlit Page Structure
```python
def show_page():
    st.title("Page Title")
    data = get_cached_data()  # Use @st.cache_data
    
    if st.button("Process"):
        with st.spinner("Processing..."):
            result = asyncio.run(some_async_operation())
```

### Bulk Operations Pattern
```python
result = await data_pipeline.process_sample_data()
# Returns: {"jobs": {"processed": 10, "errors": []}}
```
