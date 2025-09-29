# 🤖 AI Resume Matcher

**LangChain-Powered Intelligent Resume Matching System**

A professional, production-ready AI resume matching system that uses LangChain agents to intelligently parse resumes and match candidates to job descriptions. Features robust error handling, comprehensive fallback mechanisms, and no complex database setup required.

## ✨ **Key Features**

- **🌐 Professional Web Interface**: Clean Streamlit UI with tabbed navigation
- **🤖 Enhanced AI Processing**: LangChain agents with comprehensive fallbacks
- **�️ Bulletproof Parsing**: Multi-method PDF/DOCX extraction with error recovery
- **🔍 Smart Vector Search**: ChromaDB with semantic similarity matching
- **📊 Rich Analytics**: Real-time charts and insights with Plotly
- **💼 Job Management**: Professional workflow for job posting and candidate matching
- **� Intelligent Matching**: AI-powered scoring with detailed analysis
- **📈 Data Visualization**: Comprehensive reporting and candidate comparison

## 🚀 **Quick Start**

### 1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

### 2. **Set OpenAI API Key**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

### 3. **Run Application**
```bash
python run_streamlit.py
# Visit http://localhost:8501
```

## 🎮 **Usage Modes**

### **� Streamlit Web Interface (Recommended)**
```bash
python run_streamlit.py
# Or directly: streamlit run streamlit_app.py
# Visit http://localhost:8501
```
- **📄 Resume Upload**: Multi-file upload with progress tracking
- **📋 Job Management**: Add and store job descriptions  
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

## 🏗️ **Architecture**

```
📄 Resume Upload → 🤖 LangChain Parsing → 📊 Structured Data → 🔍 Vector Storage
                                                               ↓
📋 Job Description → 🤖 LangChain Analysis → 🎯 Smart Matching → 📈 Scored Results
```

**Core Components:**
- **LangChain Agents**: GPT-3.5-turbo powered intelligent parsing
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
│   ├── services/       # LangChain agents and processing
│   ├── utils/          # File handling utilities
│   └── main.py         # FastAPI server
├── data/               # File-based storage
│   ├── resumes/        # Processed resume data
│   ├── jobs/           # Job description data
│   ├── results/        # Matching results
│   └── vectordb/       # ChromaDB storage
├── demo_example.py     # Interactive demo script
├── requirements.txt    # Dependencies
└── TASKLIST.md        # Complete project status
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
OPENAI_API_KEY=your_key_here
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
