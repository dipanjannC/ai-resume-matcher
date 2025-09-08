"""
Simple data models for resume processing without database dependencies.
Uses dataclasses and Pydantic for clean, human-readable data structures.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pydantic import BaseModel
import uuid


@dataclass
class ProfileInfo:
    """Candidate profile information extracted from resume"""
    name: str = ""
    title: str = ""
    email: str = ""
    phone: str = ""
    linkedin: str = ""
    location: str = ""


@dataclass
class ExperienceInfo:
    """Work experience information"""
    total_years: int = 0
    roles: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    responsibilities: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)


@dataclass
class SkillsInfo:
    """Skills categorization"""
    technical: List[str] = field(default_factory=list)
    soft: List[str] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)


@dataclass
class TopicsInfo:
    """Domain expertise and interests"""
    domains: List[str] = field(default_factory=list)
    specializations: List[str] = field(default_factory=list)
    interests: List[str] = field(default_factory=list)


@dataclass
class ToolsLibrariesInfo:
    """Technical tools and libraries"""
    programming_languages: List[str] = field(default_factory=list)
    frameworks: List[str] = field(default_factory=list)
    tools: List[str] = field(default_factory=list)
    databases: List[str] = field(default_factory=list)
    cloud_platforms: List[str] = field(default_factory=list)


@dataclass
class ResumeData:
    """Complete structured resume data"""
    # Basic info
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    raw_text: str = ""
    processed_at: datetime = field(default_factory=datetime.now)
    
    # Structured data
    profile: ProfileInfo = field(default_factory=ProfileInfo)
    experience: ExperienceInfo = field(default_factory=ExperienceInfo)
    skills: SkillsInfo = field(default_factory=SkillsInfo)
    topics: TopicsInfo = field(default_factory=TopicsInfo)
    tools_libraries: ToolsLibrariesInfo = field(default_factory=ToolsLibrariesInfo)
    
    # AI-generated content
    summary: str = ""
    key_strengths: List[str] = field(default_factory=list)
    
    # Vector data
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "filename": self.filename,
            "raw_text": self.raw_text,
            "processed_at": self.processed_at.isoformat(),
            "profile": self.profile.__dict__,
            "experience": self.experience.__dict__,
            "skills": self.skills.__dict__,
            "topics": self.topics.__dict__,
            "tools_libraries": self.tools_libraries.__dict__,
            "summary": self.summary,
            "key_strengths": self.key_strengths,
            "has_embedding": self.embedding is not None
        }


@dataclass
class JobDescription:
    """Job description data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    company: str = ""
    raw_text: str = ""
    processed_at: datetime = field(default_factory=datetime.now)
    
    # Extracted requirements
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    experience_years: int = 0
    education_level: str = ""
    responsibilities: List[str] = field(default_factory=list)
    
    # Vector data
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "id": self.id,
            "title": self.title,
            "company": self.company,
            "raw_text": self.raw_text,
            "processed_at": self.processed_at.isoformat(),
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "experience_years": self.experience_years,
            "education_level": self.education_level,
            "responsibilities": self.responsibilities,
            "has_embedding": self.embedding is not None
        }


@dataclass
class MatchResult:
    """Result of matching a resume to a job description"""
    resume_id: str
    job_id: str
    overall_score: float
    
    # Detailed scoring
    skills_match_score: float
    experience_match_score: float
    semantic_similarity_score: float
    
    # Detailed analysis
    matching_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    strength_areas: List[str] = field(default_factory=list)
    improvement_areas: List[str] = field(default_factory=list)
    
    # AI insights
    match_summary: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "resume_id": self.resume_id,
            "job_id": self.job_id,
            "overall_score": self.overall_score,
            "skills_match_score": self.skills_match_score,
            "experience_match_score": self.experience_match_score,
            "semantic_similarity_score": self.semantic_similarity_score,
            "matching_skills": self.matching_skills,
            "missing_skills": self.missing_skills,
            "strength_areas": self.strength_areas,
            "improvement_areas": self.improvement_areas,
            "match_summary": self.match_summary,
            "recommendation": self.recommendation
        }


# Pydantic models for API validation (if needed)
class ResumeUploadRequest(BaseModel):
    filename: str
    content: str  # Base64 encoded or raw text


class JobMatchRequest(BaseModel):
    job_description: str
    job_title: Optional[str] = ""
    company: Optional[str] = ""


class MatchResponse(BaseModel):
    matches: List[Dict[str, Any]]
    total_candidates: int
    processing_time: float
