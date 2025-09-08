"""
Main orchestrator service for AI Resume Matcher.
Coordinates LangChain agents, embeddings, and vector storage without database complexity.
"""

import asyncio
import json
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
    Main service orchestrating resume processing with LangChain agents.
    Provides clean, human-readable interface for all operations.
    """
    
    def __init__(self):
        """Initialize the resume processor with all required services"""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.processed_resumes: Dict[str, ResumeData] = {}
        self.processed_jobs: Dict[str, JobDescription] = {}
        
        logger.info("ResumeProcessor initialized successfully")
    
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
            resume_data.embedding = embedding
            
            # Store in vector database
            metadata = {
                "candidate_id": resume_data.id,
                "filename": resume_data.filename,
                "skills": resume_data.skills.technical,
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
        Process resume from raw content string.
        
        Args:
            content: Raw resume text content
            filename: Associated filename
            
        Returns:
            ResumeData object with structured information
        """
        try:
            logger.info(f"Processing resume content for: {filename}")
            
            # Parse with LangChain agents
            resume_data = await langchain_agents.parse_resume(content)
            resume_data.filename = filename
            resume_data.raw_text = content
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(content)
            resume_data.embedding = embedding
            
            # Store in vector database
            metadata = {
                "candidate_id": resume_data.id,
                "filename": filename,
                "skills": resume_data.skills.technical,
                "experience_years": resume_data.experience.total_years,
                "processed_at": resume_data.processed_at.isoformat()
            }
            
            self.vector_store.add_resume(
                candidate_id=resume_data.id,
                embedding=embedding,
                document=content,
                metadata=metadata
            )
            
            # Save structured data
            await self._save_resume_data(resume_data)
            
            # Keep in memory
            self.processed_resumes[resume_data.id] = resume_data
            
            logger.info(f"Successfully processed resume content: {resume_data.id}")
            return resume_data
            
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
            job_data.embedding = embedding
            
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
                job_data.embedding = self.embedding_service.generate_embedding(job_data.raw_text)
            
            # Search vector store for similar resumes
            similar_resumes = self.vector_store.search_similar(
                query_embedding=job_data.embedding,
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


# Global processor instance
resume_processor = ResumeProcessor()
