"""
Enhanced Main orchestrator service for AI Resume Matcher.
Coordinates LangChain agents, embeddings, and vector storage with robust error handling.
"""

import asyncio
import json
import uuid
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.models.resume_data import ResumeData, JobDescription, MatchResult
from app.services.langchain_agents import langchain_agents
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.utils.file_utils import extract_text_from_file, save_uploaded_file, cleanup_temp_file
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ResumeMatcherException

logger = get_logger(__name__)


class ResumeProcessor:
    """
    Enhanced resume processor with robust error handling and fallback mechanisms.
    Provides clean, human-readable interface for all operations.
    """
    
    def __init__(self):
        """Initialize the resume processor with all required services"""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.processed_resumes: Dict[str, ResumeData] = {}
        self.processed_jobs: Dict[str, JobDescription] = {}
        
        # Initialize LangChain agents with error handling
        self.langchain_available = self._initialize_langchain_agents()
        
        logger.info("ResumeProcessor initialized successfully")
    
    def _initialize_langchain_agents(self) -> bool:
        """Initialize LangChain agents with error handling"""
        try:
            # Test if langchain_agents is working
            if hasattr(langchain_agents, 'parse_resume'):
                logger.info("LangChain agents available for AI parsing")
                return True
            else:
                logger.warning("LangChain agents not properly configured")
                return False
        except Exception as e:
            logger.warning(f"LangChain agents initialization issue: {str(e)}")
            logger.warning("AI parsing will use fallback methods")
            return False
    
    async def process_resume_file(self, file_path: str, filename: Optional[str] = None) -> ResumeData:
        """
        Process a resume file end-to-end using LangChain agents.
        
        Args:
            file_path: Path to the resume file
            filename: Optional filename override
            
        Returns:
            ResumeData object with complete structured information
        """
        try:
            logger.info(f"Starting resume processing for: {file_path}")
            
            # Extract text from file
            resume_text = extract_text_from_file(file_path)
            logger.info(f"Extracted {len(resume_text)} characters from resume")
            
            # Parse with LangChain agents
            resume_data = await langchain_agents.parse_resume(resume_text)
            resume_data.filename = filename or Path(file_path).name
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(resume_text)
            # Ensure embedding is proper format before storing
            resume_data.embedding = [float(x) for x in embedding] if embedding else None
            
            # Store in vector database with proper metadata types
            metadata = {
                "candidate_id": resume_data.id,
                "filename": resume_data.filename,
                "skills": ", ".join(resume_data.skills.technical) if resume_data.skills.technical else "",
                "skills_count": len(resume_data.skills.technical) if resume_data.skills.technical else 0,
                "experience_years": resume_data.experience.total_years,
                "processed_at": resume_data.processed_at.isoformat()
            }
            
            self.vector_store.add_resume(
                candidate_id=resume_data.id,
                embedding=embedding,
                document=resume_text,
                metadata=metadata
            )
            
            # Save structured data to file
            await self._save_resume_data(resume_data)
            
            # Keep in memory for quick access
            self.processed_resumes[resume_data.id] = resume_data
            
            logger.info(f"Successfully processed resume: {resume_data.id}")
            return resume_data
            
        except Exception as e:
            logger.error(f"Failed to process resume file {file_path}: {str(e)}")
            raise ResumeMatcherException(f"Resume processing failed: {str(e)}")
    
    async def process_resume_content(self, content: str, filename: str) -> ResumeData:
        """
        Process resume from raw content string with enhanced error handling.
        
        Args:
            content: Raw resume text content
            filename: Associated filename
            
        Returns:
            ResumeData object with structured information
        """
        try:
            logger.info(f"Processing resume content for: {filename}")
            
            # Validate content
            if not content or len(content.strip()) < 50:
                raise ResumeMatcherException(f"Resume content too short or empty: {len(content)} characters")
            
            # Parse resume with AI or fallback
            if self.langchain_available:
                try:
                    logger.info("Attempting AI-powered parsing with LangChain")
                    resume_data = await langchain_agents.parse_resume(content)
                    resume_data.filename = filename
                    resume_data.raw_text = content
                    logger.info("AI parsing successful")
                except Exception as e:
                    logger.warning(f"AI parsing failed: {str(e)}, using fallback")
                    resume_data = self._create_fallback_resume_data(content, filename)
            else:
                logger.info("Using fallback parsing (no AI available)")
                resume_data = self._create_fallback_resume_data(content, filename)
            
            # Generate embeddings
            try:
                embedding = self.embedding_service.generate_embedding(content)
                resume_data.embedding = [float(x) for x in embedding] if embedding else None
                logger.info(f"Embeddings generated: {len(embedding)} dimensions")
            except Exception as e:
                logger.warning(f"Embedding generation failed: {str(e)}")
                # Continue without embeddings - they can be generated later
                embedding = []
            
            # Store in vector database if embeddings available
            if embedding:
                try:
                    metadata = {
                        "candidate_id": resume_data.id,
                        "filename": filename,
                        "skills": ", ".join(resume_data.skills.technical) if resume_data.skills.technical else "",
                        "skills_count": len(resume_data.skills.technical) if resume_data.skills.technical else 0,
                        "experience_years": resume_data.experience.total_years,
                        "processed_at": resume_data.processed_at.isoformat()
                    }
                    
                    self.vector_store.add_resume(
                        candidate_id=resume_data.id,
                        embedding=embedding,
                        document=content,
                        metadata=metadata
                    )
                    logger.info("Resume stored in vector database")
                except Exception as e:
                    logger.warning(f"Vector storage failed: {str(e)}")
                    # Continue without vector storage
            
            # Save structured data
            try:
                await self._save_resume_data(resume_data)
                logger.info("Resume data saved to file storage")
            except Exception as e:
                logger.warning(f"File storage failed: {str(e)}")
                # Continue - we still have the parsed data
            
            # Keep in memory
            self.processed_resumes[resume_data.id] = resume_data
            
            logger.info(f"Successfully processed resume content: {resume_data.id}")
            return resume_data
            
        except ResumeMatcherException:
            raise
        except Exception as e:
            logger.error(f"Failed to process resume content: {str(e)}")
            raise ResumeMatcherException(f"Resume content processing failed: {str(e)}")
    
    async def process_job_description(self, job_text: str, title: str = "", company: str = "") -> JobDescription:
        """
        Process job description using LangChain agents.
        
        Args:
            job_text: Raw job description text
            title: Job title (optional)
            company: Company name (optional)
            
        Returns:
            JobDescription object with structured requirements
        """
        try:
            logger.info(f"Processing job description: {title}")
            
            # Parse with LangChain agents
            job_data = await langchain_agents.parse_job_description(job_text)
            
            # Override with provided info if available
            if title:
                job_data.title = title
            if company:
                job_data.company = company
            
            # Generate embedding for job description
            embedding = self.embedding_service.generate_embedding(job_text)
            job_data.embedding = [float(x) for x in embedding] if embedding else None
            
            # Save structured data
            await self._save_job_data(job_data)
            
            # Keep in memory
            self.processed_jobs[job_data.id] = job_data
            
            logger.info(f"Successfully processed job description: {job_data.id}")
            return job_data
            
        except Exception as e:
            logger.error(f"Failed to process job description: {str(e)}")
            raise ResumeMatcherException(f"Job description processing failed: {str(e)}")
    
    async def find_best_matches(self, job_data: JobDescription, top_k: int = 10) -> List[MatchResult]:
        """
        Find best resume matches for a job using vector similarity and LangChain analysis.
        
        Args:
            job_data: Structured job description
            top_k: Number of top matches to return
            
        Returns:
            List of MatchResult objects sorted by overall score
        """
        try:
            logger.info(f"Finding matches for job: {job_data.title}")
            
            # Ensure job has embedding
            if not job_data.embedding:
                logger.warning("Job description has no embedding, generating one")
                job_data.embedding = [float(x) for x in self.embedding_service.generate_embedding(job_data.raw_text)]
            
            # Search vector store for similar resumes
            # Ensure embedding is in correct format (in case loaded from disk)
            query_embedding = [float(x) for x in job_data.embedding] if job_data.embedding else []
            similar_resumes = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=min(top_k * 2, 50)  # Get more candidates for filtering
            )
            
            match_results = []
            
            # Analyze each candidate with LangChain
            for result in similar_resumes:
                try:
                    resume_id = result['candidate_id']
                    
                    # Get resume data
                    resume_data = await self._get_resume_data(resume_id)
                    if not resume_data:
                        continue
                    
                    # Analyze match with LangChain
                    match_result = await langchain_agents.analyze_match(resume_data, job_data)
                    
                    # Add semantic similarity score from vector search
                    semantic_score = result['similarity']
                    match_result.semantic_similarity_score = semantic_score
                    
                    # Recalculate overall score with semantic component
                    match_result.overall_score = (
                        match_result.skills_match_score * 0.4 +
                        match_result.experience_match_score * 0.3 +
                        semantic_score * 0.3
                    )
                    
                    match_results.append(match_result)
                    
                except Exception as e:
                    logger.warning(f"Failed to analyze match for resume {result.get('candidate_id', 'unknown')}: {str(e)}")
                    continue
            
            # Sort by overall score and return top_k
            match_results.sort(key=lambda x: x.overall_score, reverse=True)
            final_results = match_results[:top_k]
            
            # Save match results
            await self._save_match_results(job_data.id, final_results)
            
            logger.info(f"Found {len(final_results)} matches for job {job_data.title}")
            return final_results
            
        except Exception as e:
            logger.error(f"Failed to find matches: {str(e)}")
            raise ResumeMatcherException(f"Match finding failed: {str(e)}")
    
    async def get_resume_summary(self, resume_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of a processed resume"""
        try:
            resume_data = await self._get_resume_data(resume_id)
            if not resume_data:
                raise ResumeMatcherException(f"Resume {resume_id} not found")
            
            # Generate enhanced summary if needed
            if not resume_data.summary:
                resume_data.summary = await langchain_agents.generate_summary(resume_data)
                await self._save_resume_data(resume_data)
            
            return {
                "id": resume_data.id,
                "filename": resume_data.filename,
                "profile": resume_data.profile.__dict__,
                "experience_summary": {
                    "total_years": resume_data.experience.total_years,
                    "recent_roles": resume_data.experience.roles[:3],
                    "companies": resume_data.experience.companies[:3]
                },
                "skills_summary": {
                    "technical_count": len(resume_data.skills.technical),
                    "top_skills": resume_data.skills.technical[:10],
                    "certifications": resume_data.skills.certifications
                },
                "summary": resume_data.summary,
                "key_strengths": resume_data.key_strengths,
                "processed_at": resume_data.processed_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get resume summary: {str(e)}")
            raise ResumeMatcherException(f"Resume summary failed: {str(e)}")
    
    async def list_processed_resumes(self) -> List[Dict[str, Any]]:
        """List all processed resumes with basic information"""
        try:
            resumes = []
            
            # Load from file system if not in memory
            resume_files = Path(settings.RESUMES_DIR).glob("*.json")
            
            for resume_file in resume_files:
                try:
                    with open(resume_file, 'r') as f:
                        data = json.load(f)
                    
                    resumes.append({
                        "id": data["id"],
                        "filename": data["filename"],
                        "name": data.get("profile", {}).get("name", "Unknown"),
                        "title": data.get("profile", {}).get("title", ""),
                        "experience_years": data.get("experience", {}).get("total_years", 0),
                        "processed_at": data["processed_at"]
                    })
                    
                except Exception as e:
                    logger.warning(f"Could not load resume file {resume_file}: {str(e)}")
                    continue
            
            return sorted(resumes, key=lambda x: x["processed_at"], reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to list resumes: {str(e)}")
            return []
    
    async def _save_resume_data(self, resume_data: ResumeData):
        """Save resume data to file system"""
        try:
            file_path = Path(settings.RESUMES_DIR) / f"{resume_data.id}.json"
            data = resume_data.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save resume data: {str(e)}")
    
    async def _save_job_data(self, job_data: JobDescription):
        """Save job description data to file system"""
        try:
            file_path = Path(settings.JOBS_DIR) / f"{job_data.id}.json"
            data = job_data.to_dict()
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save job data: {str(e)}")
    
    async def _save_match_results(self, job_id: str, results: List[MatchResult]):
        """Save match results to file system"""
        try:
            file_path = Path(settings.RESULTS_DIR) / f"matches_{job_id}.json"
            data = {
                "job_id": job_id,
                "generated_at": datetime.now().isoformat(),
                "total_matches": len(results),
                "matches": [result.to_dict() for result in results]
            }
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save match results: {str(e)}")
    
    async def _get_resume_data(self, resume_id: str) -> Optional[ResumeData]:
        """Get resume data from memory or file system"""
        try:
            # Check memory first
            if resume_id in self.processed_resumes:
                return self.processed_resumes[resume_id]
            
            # Load from file system
            file_path = Path(settings.RESUMES_DIR) / f"{resume_id}.json"
            if not file_path.exists():
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert back to ResumeData object (simplified)
            resume_data = ResumeData(
                id=data["id"],
                filename=data["filename"],
                raw_text=data["raw_text"],
                summary=data["summary"],
                key_strengths=data["key_strengths"]
            )
            
            # Load structured data
            if "profile" in data:
                resume_data.profile = resume_data.profile.__class__(**data["profile"])
            if "experience" in data:
                resume_data.experience = resume_data.experience.__class__(**data["experience"])
            if "skills" in data:
                resume_data.skills = resume_data.skills.__class__(**data["skills"])
            if "topics" in data:
                resume_data.topics = resume_data.topics.__class__(**data["topics"])
            if "tools_libraries" in data:
                resume_data.tools_libraries = resume_data.tools_libraries.__class__(**data["tools_libraries"])
            
            self.processed_resumes[resume_id] = resume_data
            return resume_data
            
        except Exception as e:
            logger.error(f"Failed to get resume data for {resume_id}: {str(e)}")
            return None
    
    def _create_fallback_resume_data(self, content: str, filename: str) -> ResumeData:
        """
        Create resume data using rule-based parsing as fallback.
        
        Args:
            content: Resume text content
            filename: Original filename
            
        Returns:
            ResumeData object with basic parsed information
        """
        logger.info("Creating fallback resume data using rule-based parsing")
        
        try:
            # Basic rule-based extraction
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Extract basic profile information
            name = self._extract_name(lines)
            email = self._extract_email(content)
            phone = self._extract_phone(content)
            
            # Extract skills
            technical_skills = self._extract_technical_skills(content)
            
            # Create basic resume data structure
            from app.models.resume_data import (
                ProfileInfo, ExperienceInfo, SkillsInfo, 
                TopicsInfo, ToolsLibrariesInfo
            )
            
            resume_data = ResumeData(
                id=str(uuid.uuid4()),
                raw_text=content,
                filename=filename,
                profile=ProfileInfo(
                    name=name or "Unknown",
                    title=self._extract_title(lines),
                    email=email,
                    phone=phone,
                    linkedin=self._extract_linkedin(content),
                    location=self._extract_location(content)
                ),
                experience=ExperienceInfo(
                    total_years=self._estimate_experience_years(content),
                    roles=self._extract_roles(content),
                    companies=self._extract_companies(content),
                    responsibilities=[],
                    achievements=[]
                ),
                skills=SkillsInfo(
                    technical=technical_skills,
                    soft=[],
                    certifications=[],
                    languages=[]
                ),
                topics=TopicsInfo(
                    domains=[],
                    specializations=[],
                    interests=[]
                ),
                tools_libraries=ToolsLibrariesInfo(
                    programming_languages=self._extract_programming_languages(content),
                    frameworks=[],
                    tools=[],
                    databases=[],
                    cloud_platforms=[]
                ),
                summary=f"Resume processed from {filename} using fallback parsing. Manual review recommended.",
                key_strengths=["Experience extraction needed", "Skills review needed", "Manual verification recommended"],
                processed_at=datetime.now()
            )
            
            logger.info("Fallback resume data created successfully")
            return resume_data
            
        except Exception as e:
            logger.error(f"Fallback parsing failed: {str(e)}")
            # Create absolute minimal structure
            return self._create_minimal_resume_data(content, filename)
    
    def _create_minimal_resume_data(self, content: str, filename: str) -> ResumeData:
        """Create minimal resume data when all parsing fails"""
        from app.models.resume_data import (
            ProfileInfo, ExperienceInfo, SkillsInfo, 
            TopicsInfo, ToolsLibrariesInfo
        )
        
        return ResumeData(
            id=str(uuid.uuid4()),
            raw_text=content,
            filename=filename,
            profile=ProfileInfo(
                name="Parsing Failed",
                title="Unknown",
                email="",
                phone="",
                linkedin="",
                location=""
            ),
            experience=ExperienceInfo(
                total_years=0,
                roles=[],
                companies=[],
                responsibilities=[],
                achievements=[]
            ),
            skills=SkillsInfo(
                technical=[],
                soft=[],
                certifications=[],
                languages=[]
            ),
            topics=TopicsInfo(
                domains=[],
                specializations=[],
                interests=[]
            ),
            tools_libraries=ToolsLibrariesInfo(
                programming_languages=[],
                frameworks=[],
                tools=[],
                databases=[],
                cloud_platforms=[]
            ),
            summary=f"Resume from {filename} requires manual processing. Automated parsing failed.",
            key_strengths=["Manual review required"],
            processed_at=datetime.now()
        )
    
    # Rule-based extraction methods
    def _extract_name(self, lines: list) -> str:
        """Extract name from resume lines"""
        # Look for common name patterns in first few lines
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
                
            # Skip common headers
            if any(word.lower() in line.lower() for word in ['resume', 'cv', 'curriculum', 'profile', 'contact']):
                continue
                
            # Look for name patterns (2-4 words, proper case)
            name_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]*){1,3})(?:\s|$)'
            match = re.match(name_pattern, line)
            if match:
                potential_name = match.group(1).strip()
                # Avoid job titles and common words
                if not any(word.lower() in potential_name.lower() for word in ['engineer', 'developer', 'manager', 'analyst', 'consultant', 'senior', 'junior', 'lead']):
                    return potential_name
        
        return ""
    
    def _extract_email(self, text: str) -> str:
        """Extract email address"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group() if match else ""
    
    def _extract_phone(self, text: str) -> str:
        """Extract phone number"""
        phone_pattern = r'[\+]?[1-9]?[0-9]{3}[-\s\.]?[0-9]{3}[-\s\.]?[0-9]{4,6}'
        match = re.search(phone_pattern, text)
        return match.group() if match else ""
    
    def _extract_title(self, lines: list) -> str:
        """Extract job title"""
        common_titles = [
            'engineer', 'developer', 'manager', 'analyst', 
            'consultant', 'specialist', 'coordinator', 'director',
            'scientist', 'architect', 'lead', 'senior', 'junior',
            'intern', 'associate', 'principal', 'staff'
        ]
        
        # Look for title in first several lines, often after name
        for i, line in enumerate(lines[:15]):
            line_clean = line.strip()
            if not line_clean:
                continue
                
            line_lower = line_clean.lower()
            
            # Skip email, phone, linkedin lines
            if '@' in line or 'linkedin' in line_lower or re.search(r'\+?\d{10,}', line):
                continue
                
            # Check for title keywords
            if any(title in line_lower for title in common_titles):
                return line_clean
                
            # Look for "ML Engineer", "Data Scientist" etc patterns
            title_patterns = [
                r'(ml|machine learning|data|software|backend|frontend|full.?stack|devops|cloud)\s+(engineer|developer|scientist|analyst)',
                r'(senior|junior|lead|principal|staff)\s+(engineer|developer|scientist|analyst|manager)',
                r'(product|project|program)\s+manager'
            ]
            
            for pattern in title_patterns:
                if re.search(pattern, line_lower):
                    return line_clean
        
        return ""
    
    def _extract_linkedin(self, text: str) -> str:
        """Extract LinkedIn URL"""
        linkedin_pattern = r'linkedin\.com/in/[\w-]+'
        match = re.search(linkedin_pattern, text)
        return match.group() if match else ""
    
    def _extract_location(self, text: str) -> str:
        """Extract location"""
        # Simple location extraction
        location_pattern = r'[A-Z][a-z]+,\s*[A-Z]{2}'
        match = re.search(location_pattern, text)
        return match.group() if match else ""
    
    def _extract_technical_skills(self, text: str) -> list:
        """Extract technical skills"""
        common_skills = [
            # Programming languages
            'python', 'javascript', 'java', 'c++', 'c#', 'typescript', 'golang', 'rust', 'scala',
            'r', 'matlab', 'swift', 'kotlin', 'php', 'ruby', 'perl',
            
            # Web frameworks
            'react', 'angular', 'vue', 'django', 'flask', 'fastapi', 'express', 'spring', 'nodejs',
            
            # Databases
            'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'solr', 'neo4j',
            'cassandra', 'sqlite', 'oracle', 'chromadb',
            
            # Cloud and DevOps
            'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git', 'github', 'gitlab',
            'terraform', 'ansible', 'helm', 'docker-compose',
            
            # ML/AI
            'pytorch', 'tensorflow', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
            'keras', 'xgboost', 'lightgbm', 'opencv', 'nltk', 'spacy', 'transformers', 'langchain',
            
            # Big Data
            'spark', 'hadoop', 'kafka', 'airflow', 'dbt', 'snowflake', 'databricks',
            
            # Other tools
            'jupyter', 'vscode', 'intellij', 'postman', 'jira', 'confluence'
        ]
        
        found_skills = []
        text_lower = text.lower()
        
        # Use word boundaries to avoid partial matches
        for skill in common_skills:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                # Preserve original case for known acronyms
                if skill.upper() in ['SQL', 'AWS', 'GCP', 'API', 'ML', 'AI', 'GPU', 'CPU', 'REST']:
                    found_skills.append(skill.upper())
                else:
                    found_skills.append(skill.title())
                
        return list(set(found_skills))  # Remove duplicates
    
    def _extract_programming_languages(self, text: str) -> list:
        """Extract programming languages"""
        languages = [
            'python', 'javascript', 'java', 'c++', 'c#', 'ruby',
            'php', 'swift', 'kotlin', 'go', 'rust', 'typescript'
        ]
        
        found_languages = []
        text_lower = text.lower()
        
        for lang in languages:
            if lang in text_lower:
                found_languages.append(lang.title())
                
        return found_languages
    
    def _extract_roles(self, text: str) -> list:
        """Extract job roles/titles"""
        # Simple role extraction - could be enhanced
        return []
    
    def _extract_companies(self, text: str) -> list:
        """Extract company names"""
        # Simple company extraction - could be enhanced
        return []
    
    def _estimate_experience_years(self, text: str) -> int:
        """Estimate years of experience"""
        # Look for year patterns
        year_patterns = [
            r'(\d+)\s*years?\s*of\s*experience',
            r'(\d+)\s*\+?\s*years?',
            r'(\d{4})\s*-\s*(\d{4})',
            r'(\d{4})\s*-\s*present'
        ]
        
        text_lower = text.lower()
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                if isinstance(matches[0], tuple):
                    # Year range
                    start_year = int(matches[0][0])
                    end_year = int(matches[0][1]) if matches[0][1] else datetime.now().year
                    return max(0, end_year - start_year)
                else:
                    # Direct years mention
                    return int(matches[0])
        
        return 0


# Global processor instance
resume_processor = ResumeProcessor()
