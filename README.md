# 🤖 AI Resume Matcher

**LangChain-Powered Intelligent Resume Matching System**

A professional, production-ready AI resume matching system that uses LangChain agents to intelligently parse resumes and match candidates to job descriptions. Features robust error handling, comprehensive fallback mechanisms, and no complex database setup required.

## ✨ **Key Features**

- **🌐 Professional Web Interface**: Clean Streamlit UI with sidebar navigation for core actions
- **🤖 Multi-LLM Support**: Robust fallback system using Groq, Gemini, and OpenAI
- **🧠 Advanced Memory**: Integrated Mem0 and Graphiti for context-aware interactions
- **️ Bulletproof Parsing**: Multi-method PDF/DOCX extraction with error recovery
- **🔍 Smart Vector Search**: ChromaDB with semantic similarity matching
- **📊 Rich Analytics**: Real-time charts and insights with Plotly
- **💼 Job Management**: Professional workflow for job posting and candidate matching
- ** Intelligent Matching**: AI-powered scoring with detailed analysis
- **📈 Data Visualization**: Comprehensive reporting and candidate comparison

## 🚀 **Quick Start**

### 1. **Install Dependencies**
Using `uv` (recommended):
```bash
uv pip install -r requirements.txt
```
Or with pip:
```bash
pip install -r requirements.txt
```

### 2. **Set API Keys**
You can set these in a `.env` file or configure them in the Streamlit Sidebar.
```bash
export GROQ_API_KEY=your_groq_key
export GEMINI_API_KEY=your_gemini_key
export OPENAI_API_KEY=your_openai_key  # Optional fallback
```

### 3. **Run Application**
```bash
streamlit run streamlit_app.py
# Visit http://localhost:8501
```

## 🎮 **Usage Modes**

### ** Streamlit Web Interface (Recommended)**
```bash
streamlit run streamlit_app.py
```
- **📄 Resume Upload**: Sidebar-based multi-file upload
- **📋 Job Management**: Add jobs via sidebar (URL extraction or manual)
- **🎯 Interactive Matching**: Real-time candidate-job matching
- **📊 Analytics Dashboard**: Visualizations and insights
- **🔍 Advanced Search**: Filter and search candidates

### **🎯 Interactive Demo (CLI)**
```bash
python quick_demo.py
```
- Upload and process resumes
- Analyze job descriptions  
- Find matching candidates
- View detailed AI insights

### **🖥️ CLI Mode**
```bash
python app/main.py cli
```

### **🌐 Web Server Mode**
```bash
python app/main.py
# Visit http://localhost:8000
```

### **🧪 Running Tests (TDD)**
The project includes a comprehensive, robust test suite for the LangChain parsing engines and customizer:
```bash
# Run the entire test suite
uv run pytest tests/ -v
```

## 🏗️ **Architecture**

```
📄 Resume Upload → 🤖 LangChain Agents (Groq/Gemini/OpenAI) → 🧠 Memory (Mem0/Graphiti)
                                      ↓
📋 Job Description → 🤖 Analysis → 🎯 Smart Matching → 📈 Scored Results
```

**Core Components:**
- **LangChain Agents**: Multi-provider LLM orchestration with automatic fallback
- **Memory Layer**: Mem0 and Graphiti for long-term context
- **ChromaDB**: Vector database for semantic similarity
- **File Storage**: Simple JSON-based data persistence
- **FastAPI**: Clean REST API interface

## 📊 **What Gets Extracted**

### **From Resumes:**
- **👤 Profile**: Name, title, contact, location
- **💼 Experience**: Years, roles, companies, achievements
- **🛠️ Skills**: Technical, soft, certifications
- **📚 Topics**: Domains, specializations, interests
- **🔧 Tools**: Languages, frameworks, databases, cloud platforms
- **✨ AI Summary**: Generated professional summary

### **From Job Descriptions:**
- **📋 Requirements**: Required vs preferred skills
- **⏰ Experience**: Years and level requirements
- **🎯 Responsibilities**: Key job duties
- **🏢 Company**: Company and role information

### **Matching Results:**
- **📊 Detailed Scoring**: Skills, experience, semantic similarity
- **✅ Skill Analysis**: Matching and missing skills
- **💪 Strengths**: Candidate strength areas
- **📈 Recommendations**: AI-powered hiring insights

## 📁 **Project Structure**

```
ai-resume-matcher/
├── app/
│   ├── core/           # Configuration and utilities
│   ├── models/         # Data models (no database)
│   ├── services/       # LangChain agents, Memory, and processing
│   ├── utils/          # File handling utilities
│   └── main.py         # FastAPI server
├── data/               # File-based storage
│   ├── resumes/        # Processed resume data
│   ├── jobs/           # Job description data
│   ├── results/        # Matching results
│   └── vectordb/       # ChromaDB storage
├── demo_example.py     # Interactive demo script
├── requirements.txt    # Dependencies
└── task.md             # Complete project status
```

## 🛠️ **API Endpoints**

- `POST /upload-resume` - Upload and process resume files
- `POST /match-job` - Find matching candidates for job
- `GET /resumes` - List all processed resumes  
- `GET /resumes/{id}` - Get detailed resume analysis
- `GET /health` - Health check

## 🔧 **Configuration**

Create a `.env` file:
```env
# LLM Providers (At least one required)
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
OPENAI_API_KEY=your_openai_key

# Memory (Optional)
MEMORY_PROVIDER=mem0  # or graphiti
MEM0_API_KEY=your_mem0_key
GRAPHITI_URL=your_graphiti_url
GRAPHITI_API_KEY=your_graphiti_key

# System
DEBUG=false
LOG_LEVEL=INFO
MAX_FILE_SIZE_MB=10
```

## 📝 **Example Usage**

### **Process a Resume:**
```python
from app.services.resume_processor import resume_processor

# Process resume file
resume_data = await resume_processor.process_resume_file("resume.pdf")

# Get structured data
print(f"Candidate: {resume_data.profile.name}")
print(f"Experience: {resume_data.experience.total_years} years")
print(f"Skills: {resume_data.skills.technical}")
```

### **Find Job Matches:**
```python
# Process job description
job_data = await resume_processor.process_job_description(
    job_text, "Senior Developer", "TechCorp"
)

# Find matches
matches = await resume_processor.find_best_matches(job_data, top_k=5)

# View results
for match in matches:
    print(f"Score: {match.overall_score:.2f}")
    print(f"Recommendation: {match.recommendation}")
```

## 🎯 **Why This Approach?**

### **✅ Advantages:**
- **Simple Setup**: No database configuration required
- **AI-Powered**: LangChain agents provide intelligent analysis
- **Human-Readable**: Clean, documented, maintainable code
- **Flexible**: Easy to customize and extend
- **Fast**: File-based storage with vector search

### **🎯 Perfect For:**
- Startups and small teams
- Proof of concepts and demos
- Educational projects
- Rapid prototyping
- Simple recruitment workflows

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## 📄 **License**

MIT License - see LICENSE file for details.

## 🙋 **Support**

- 📖 Check `TASKLIST.md` for detailed project status
- 🎮 Run `python demo_example.py` for interactive demo
- 🐛 Open issues for bugs or feature requests

---

**Built with ❤️ using LangChain, OpenAI, and modern Python**
