# AI Resume Matcher - Copilot Instructions

## Architecture Overview

This is a **LangChain-powered resume matching system** using file-based storage (no traditional database). The core pattern is:
- **LangChain agents** parse unstructured resumes/jobs into structured data using OpenAI GPT-3.5-turbo
- **ChromaDB** provides vector storage with multi-collection semantic search
- **Streamlit** serves the interactive web interface with caching
- **FastAPI** provides REST API endpoints
- **File-based persistence** in `./data/` directory (JSON files)

## Key Components & Service Boundaries

### Core Services (`app/services/`)
- `resume_processor.py` - Main orchestrator, coordinates all resume operations
- `langchain_agents.py` - LLM-powered parsing using structured prompts and Pydantic models
- `vector_store.py` - ChromaDB wrapper for semantic similarity search
- `job_processor.py` - Job description processing and storage
- `data_pipeline.py` - Bulk operations for CSV/JSON uploads

### Data Flow Pattern
```
File Upload → Text Extraction → LangChain Agent → Structured Data → Vector Embedding → ChromaDB + JSON Storage
```

## Development Workflows

### Running the Application
```bash
# Streamlit web interface (primary interface)
python run_streamlit.py

# FastAPI server
python app/main.py

# Interactive CLI demo
python quick_demo.py

# Bulk testing
python test_job_matching.py
```

### Environment Setup
- Set `OPENAI_API_KEY` in environment or `.env` file
- The system uses Groq by default in `langchain_agents.py` (see line 39: `self.llm = LLMService().get_groq()`)
- Data persists in `./data/` with subdirectories: `resumes/`, `jobs/`, `vectordb/`, `results/`

## Project-Specific Patterns

### LangChain Integration Pattern
- **Structured Output**: All LangChain agents use Pydantic models (`app/models/langchain_models.py`) for consistent parsing
- **Prompt Management**: Centralized in `prompt_manager.py` - modify prompts here, not in agent code
- **JSON Cleaning**: `_clean_json_response()` method handles LLM output sanitization (see `langchain_agents.py:55`)

### Data Models & Storage
- **Pydantic Models**: `app/models/resume_data.py` defines all data structures
- **File-based Storage**: Structured data saved as JSON in `./data/resumes/` and `./data/jobs/`
- **Vector Metadata**: ChromaDB stores both embeddings and searchable metadata (skills, experience years, etc.)

### Streamlit Caching Strategy
```python
@st.cache_data(ttl=300)  # 5-minute cache
def get_stored_jobs():
    return asyncio.run(job_processor.list_stored_jobs())
```
Always use `st.cache_data.clear()` after bulk operations to refresh UI data.

### Error Handling Convention
- Custom exceptions in `app/core/exceptions.py`
- Structured logging with `get_logger(__name__)` pattern
- Async error handling with try/catch wrapping service calls

## Integration Points & Dependencies

### Vector Search Architecture
- **ChromaDB Collections**: Single collection `resume_embeddings` with rich metadata
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (CPU-optimized)
- **Similarity Search**: Uses cosine similarity with metadata filtering

### File Processing Pipeline
- **Supported Formats**: PDF, DOCX, TXT (configured in `config.py`)
- **Text Extraction**: `file_utils.py` handles different file types
- **Temporary Storage**: Files processed in `./data/temp/` then cleaned up

### Bulk Operations Pattern
```python
# Data pipeline pattern for bulk uploads
result = await data_pipeline.process_sample_data()
# Returns: {"jobs": {"processed": 10, "errors": []}}
```

## Testing & Debugging

### Key Test Files
- `quick_demo.py` - Interactive feature demonstration
- `test_job_matching.py` - End-to-end matching workflow
- `test_vector_search.py` - Vector similarity verification

### Debugging Vector Search
```python
# Check ChromaDB collections
candidates = vector_store.get_all_candidates()
print(f"Found {len(candidates)} stored resumes")

# Verify embeddings
stats = data_pipeline.get_pipeline_stats()
```

### Configuration Debugging
- All settings in `app/core/config.py` using Pydantic settings
- Check `settings.OPENAI_API_KEY` availability before running LangChain operations
- ChromaDB persists to `./data/vectordb/` - check this directory for storage issues

## Common Patterns

### Adding New LangChain Agents
1. Define Pydantic output model in `langchain_models.py`
2. Add prompt template to `prompt_manager.py`
3. Implement agent method in `langchain_agents.py` using `_clean_json_response()`
4. Add service method in appropriate processor class

### Streamlit Page Structure
```python
# Standard page pattern
def show_page():
    st.title("Page Title")
    
    # Use cached data loading
    data = get_cached_data()
    
    # Async operations in try/catch
    if st.button("Process"):
        with st.spinner("Processing..."):
            result = asyncio.run(some_async_operation())
```

### File Upload Handling
```python
# Standard file processing pattern
temp_path = save_uploaded_file(content, filename)
try:
    result = await processor.process_file(temp_path)
finally:
    cleanup_temp_file(temp_path)
```