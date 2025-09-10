"""
Job Processor Service - Handles job description storage and retrieval
"""

import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.models.resume_data import JobDescription
from app.services.langchain_agents import langchain_agents
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ResumeMatcherException

logger = get_logger(__name__)


class JobProcessor:
    """
    Service for processing and storing job descriptions in vector store.
    Provides job management and candidate matching capabilities.
    """
    
    def __init__(self):
        """Initialize job processor with required services"""
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.jobs_collection = "job_descriptions"
        self.stored_jobs: Dict[str, JobDescription] = {}
        
        # Ensure jobs directory exists
        self.jobs_dir = Path(settings.DATA_DIR) / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("JobProcessor initialized successfully")
    
    async def process_and_store_job(
        self,
        job_text: str,
        title: str = "",
        company: str = "",
        experience_years: int = 0,
        location: str = ""
    ) -> JobDescription:
        """
        Process job description with LangChain and store in vector database.
        
        Args:
            job_text: Raw job description text
            title: Job title
            company: Company name
            experience_years: Required experience in years
            location: Job location
            
        Returns:
            JobDescription object with structured data
        """
        try:
            logger.info(f"Processing job description: {title}")
            
            # Parse with LangChain agents
            job_data = await langchain_agents.parse_job_description(job_text)
            
            # Override with provided information
            if title:
                job_data.title = title
            if company:
                job_data.company = company
            if experience_years:
                job_data.experience_years = experience_years
            if location:
                job_data.location = location
            
            job_data.raw_text = job_text
            
            # Generate embedding
            embedding = self.embedding_service.generate_embedding(job_text)
            job_data.embedding = embedding
            
            # Store in vector database with proper metadata types
            metadata = {
                "job_id": job_data.id,
                "title": job_data.title,
                "company": job_data.company,
                "experience_years": job_data.experience_years,
                "location": job_data.location,
                "required_skills": ", ".join(job_data.required_skills) if job_data.required_skills else "",
                "skills_count": len(job_data.required_skills) if job_data.required_skills else 0,
                "created_at": job_data.created_at.isoformat()
            }
            
            # Initialize jobs collection if needed
            self._ensure_jobs_collection()
            
            # Add to vector store
            self.vector_store.add_document(
                collection_name=self.jobs_collection,
                document_id=job_data.id,
                embedding=embedding,
                document=job_text,
                metadata=metadata
            )
            
            # Save to file
            await self._save_job_data(job_data)
            
            # Keep in memory
            self.stored_jobs[job_data.id] = job_data
            
            logger.info(f"Successfully stored job: {job_data.id}")
            return job_data
            
        except Exception as e:
            logger.error(f"Failed to process job description: {str(e)}")
            raise ResumeMatcherException(f"Job processing failed: {str(e)}")
    
    async def find_candidates_for_job(self, job_id: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Find top candidates for a specific job using vector similarity.
        
        Args:
            job_id: Job identifier
            top_k: Number of top candidates to return
            
        Returns:
            List of candidate matches with scores
        """
        try:
            # Get job data
            job_data = await self.get_job_data(job_id)
            if not job_data:
                raise ResumeMatcherException(f"Job {job_id} not found")
            
            if not job_data.embedding:
                raise ResumeMatcherException(f"Job {job_id} has no embedding data")
            
            logger.info(f"Finding candidates for job: {job_data.title}")
            
            # Search for similar resumes using job embedding
            similar_resumes = self.vector_store.search_similar(
                query_embedding=job_data.embedding,
                top_k=top_k,
                collection_name="resumes"  # Search in resumes collection
            )
            
            # Enhance results with additional data
            candidates = []
            for result in similar_resumes:
                candidate_info = {
                    "candidate_id": result["candidate_id"],
                    "similarity_score": result["similarity"],
                    "metadata": result.get("metadata", {}),
                    "document_preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"]
                }
                candidates.append(candidate_info)
            
            logger.info(f"Found {len(candidates)} candidates for job {job_data.title}")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to find candidates for job {job_id}: {str(e)}")
            raise ResumeMatcherException(f"Candidate search failed: {str(e)}")
    
    async def get_job_data(self, job_id: str) -> Optional[JobDescription]:
        """
        Retrieve job data by ID.
        
        Args:
            job_id: Job identifier
            
        Returns:
            JobDescription object or None if not found
        """
        try:
            # Check memory cache first
            if job_id in self.stored_jobs:
                return self.stored_jobs[job_id]
            
            # Load from file
            job_file = self.jobs_dir / f"{job_id}.json"
            if job_file.exists():
                with open(job_file, 'r') as f:
                    job_dict = json.load(f)
                
                job_data = JobDescription.from_dict(job_dict)
                self.stored_jobs[job_id] = job_data
                return job_data
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get job data for {job_id}: {str(e)}")
            return None
    
    async def list_stored_jobs(self) -> List[Dict[str, Any]]:
        """
        List all stored jobs with summary information.
        
        Returns:
            List of job summaries
        """
        try:
            jobs = []
            
            # Load all job files
            for job_file in self.jobs_dir.glob("*.json"):
                try:
                    with open(job_file, 'r') as f:
                        job_dict = json.load(f)
                    
                    job_summary = {
                        "id": job_dict["id"],
                        "title": job_dict["title"],
                        "company": job_dict["company"],
                        "experience_years": job_dict.get("experience_years", 0),
                        "location": job_dict.get("location", ""),
                        "required_skills": job_dict.get("required_skills", []),
                        "created_at": job_dict["created_at"]
                    }
                    jobs.append(job_summary)
                
                except Exception as e:
                    logger.warning(f"Failed to load job file {job_file}: {str(e)}")
                    continue
            
            # Sort by creation date
            jobs.sort(key=lambda x: x["created_at"], reverse=True)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to list stored jobs: {str(e)}")
            return []
    
    async def delete_job(self, job_id: str) -> bool:
        """
        Delete a job from storage and vector store.
        
        Args:
            job_id: Job identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Remove from vector store
            self.vector_store.delete_document(self.jobs_collection, job_id)
            
            # Remove from file system
            job_file = self.jobs_dir / f"{job_id}.json"
            if job_file.exists():
                job_file.unlink()
            
            # Remove from memory
            if job_id in self.stored_jobs:
                del self.stored_jobs[job_id]
            
            logger.info(f"Successfully deleted job: {job_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete job {job_id}: {str(e)}")
            return False
    
    async def update_job(self, job_id: str, updates: Dict[str, Any]) -> Optional[JobDescription]:
        """
        Update an existing job description.
        
        Args:
            job_id: Job identifier
            updates: Dictionary of fields to update
            
        Returns:
            Updated JobDescription object or None if not found
        """
        try:
            job_data = await self.get_job_data(job_id)
            if not job_data:
                return None
            
            # Apply updates
            for field, value in updates.items():
                if hasattr(job_data, field):
                    setattr(job_data, field, value)
            
            # Regenerate embedding if text changed
            if "raw_text" in updates or "description" in updates:
                embedding = self.embedding_service.generate_embedding(job_data.raw_text)
                job_data.embedding = embedding
                
                # Update vector store
                metadata = {
                    "job_id": job_data.id,
                    "title": job_data.title,
                    "company": job_data.company,
                    "experience_years": job_data.experience_years,
                    "location": job_data.location,
                    "required_skills": job_data.required_skills,
                    "updated_at": datetime.now().isoformat()
                }
                
                self.vector_store.update_document(
                    collection_name=self.jobs_collection,
                    document_id=job_id,
                    embedding=embedding,
                    document=job_data.raw_text,
                    metadata=metadata
                )
            
            # Save updated data
            await self._save_job_data(job_data)
            
            # Update memory cache
            self.stored_jobs[job_id] = job_data
            
            logger.info(f"Successfully updated job: {job_id}")
            return job_data
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {str(e)}")
            return None
    
    async def search_jobs(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search jobs by text query using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of matching jobs with scores
        """
        try:
            # Generate embedding for query
            query_embedding = self.embedding_service.generate_embedding(query)
            
            # Search jobs collection
            results = self.vector_store.search_similar(
                query_embedding=query_embedding,
                top_k=top_k,
                collection_name=self.jobs_collection
            )
            
            # Format results
            jobs = []
            for result in results:
                job_info = {
                    "job_id": result["job_id"],
                    "similarity_score": result["similarity"],
                    "metadata": result.get("metadata", {}),
                    "title": result.get("metadata", {}).get("title", ""),
                    "company": result.get("metadata", {}).get("company", ""),
                    "preview": result["document"][:200] + "..." if len(result["document"]) > 200 else result["document"]
                }
                jobs.append(job_info)
            
            return jobs
            
        except Exception as e:
            logger.error(f"Failed to search jobs: {str(e)}")
            return []
    
    def _ensure_jobs_collection(self):
        """Ensure jobs collection exists in vector store"""
        try:
            # This will create the collection if it doesn't exist
            self.vector_store.get_or_create_collection(self.jobs_collection)
        except Exception as e:
            logger.warning(f"Could not ensure jobs collection: {str(e)}")
    
    async def _save_job_data(self, job_data: JobDescription):
        """Save job data to JSON file"""
        try:
            job_file = self.jobs_dir / f"{job_data.id}.json"
            
            with open(job_file, 'w') as f:
                json.dump(job_data.to_dict(), f, indent=2, default=str)
            
            logger.debug(f"Saved job data to {job_file}")
            
        except Exception as e:
            logger.error(f"Failed to save job data: {str(e)}")
            raise


# Global instance
job_processor = JobProcessor()
