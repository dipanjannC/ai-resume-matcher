"""
Pydantic models for LangChain output parsing.
These models define the structure for AI agent outputs.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any


class ResumeParsingOutput(BaseModel):
    """Structured output for resume parsing"""
    profile: Dict[str, str] = Field(description="Profile information: name, title, email, phone, linkedin, location")
    experience: Dict[str, Any] = Field(description="Experience: total_years, roles, companies, responsibilities, achievements")
    skills: Dict[str, List[str]] = Field(description="Skills: technical, soft, certifications, languages")
    topics: Dict[str, List[str]] = Field(description="Topics: domains, specializations, interests")
    tools_libraries: Dict[str, List[str]] = Field(description="Tools: programming_languages, frameworks, tools, databases, cloud_platforms")
    summary: str = Field(description="Concise professional summary")
    key_strengths: List[str] = Field(description="Top 3-5 key strengths")


class JobParsingOutput(BaseModel):
    """Structured output for job description parsing"""
    title: str = Field(description="Job title", default="Unknown Position")
    company: str = Field(description="Company name", default="Unknown Company")
    required_skills: List[str] = Field(description="Must-have skills", default_factory=list)
    preferred_skills: List[str] = Field(description="Nice-to-have skills", default_factory=list)
    experience_years: int = Field(description="Required years of experience", default=0)
    education_level: str = Field(description="Required education level", default="Not specified")
    responsibilities: List[str] = Field(description="Key responsibilities", default_factory=list)
    requirements: List[str] = Field(description="All job requirements", default_factory=list)
    company_info: str = Field(description="Company information", default="")
    summary: str = Field(description="Job summary", default="")


class MatchAnalysisOutput(BaseModel):
    """Structured output for match analysis"""
    skills_match_score: float = Field(description="Skills matching score (0-1)")
    experience_match_score: float = Field(description="Experience matching score (0-1)")
    overall_score: float = Field(description="Overall match score (0-1)")
    matching_skills: List[str] = Field(description="Skills that match")
    missing_skills: List[str] = Field(description="Required skills candidate lacks")
    strength_areas: List[str] = Field(description="Candidate's strength areas")
    improvement_areas: List[str] = Field(description="Areas for improvement")
    match_summary: str = Field(description="Brief match summary")
    recommendation: str = Field(description="Hiring recommendation")
