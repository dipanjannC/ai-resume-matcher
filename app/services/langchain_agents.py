"""
Enhanced LangChain agents for intelligent resume and job description processing.
Focused on clean, human-readable AI-powered parsing and analysis.
"""

import asyncio
from typing import Dict, Any, List, Optional
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.models.resume_data import (
    ResumeData, JobDescription, MatchResult,
    ProfileInfo, ExperienceInfo, SkillsInfo, 
    TopicsInfo, ToolsLibrariesInfo
)
from app.models.langchain_models import (
    ResumeParsingOutput, JobParsingOutput, MatchAnalysisOutput
)
from app.services.llm import LLMService
from app.services.prompt_manager import prompt_manager
from app.core.logging import get_logger
from app.core.exceptions import ResumeMatcherException

logger = get_logger(__name__)


class LangChainAgents:
    """
    Enhanced LangChain agents for intelligent resume processing.
    Provides human-readable, AI-powered analysis and matching.
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize LangChain agents with OpenAI model"""
        # Initialize LLM service - use Groq since we have the API key
        self.llm = LLMService().get_groq()  # Use Groq instead of OpenAI
        
        # Initialize parsers
        self.resume_parser = PydanticOutputParser(pydantic_object=ResumeParsingOutput)
        self.job_parser = PydanticOutputParser(pydantic_object=JobParsingOutput)
        self.match_parser = PydanticOutputParser(pydantic_object=MatchAnalysisOutput)
        
        # Get prompt templates from prompt manager
        self.resume_prompt = prompt_manager.get_resume_parsing_prompt()
        self.job_prompt = prompt_manager.get_job_parsing_prompt()
        self.match_prompt = prompt_manager.get_matching_prompt()
        self.summary_prompt = prompt_manager.get_summary_prompt()
        
        logger.info("LangChain agents initialized successfully")
    
    async def parse_resume(self, resume_text: str) -> ResumeData:
        """
        Parse resume using LangChain agent and return structured data.
        
        Args:
            resume_text: Raw resume text
            
        Returns:
            ResumeData object with structured information
        """
        try:
            logger.info("Starting resume parsing with LangChain agent")
            
            # Create chain
            chain = self.resume_prompt | self.llm | self.resume_parser
            
            # Parse resume
            result = await chain.ainvoke({
                "resume_text": resume_text,
                "format_instructions": self.resume_parser.get_format_instructions()
            })
            
            # Convert to ResumeData object
            resume_data = ResumeData(
                raw_text=resume_text,
                profile=ProfileInfo(**result.profile),
                experience=ExperienceInfo(**result.experience),
                skills=SkillsInfo(**result.skills),
                topics=TopicsInfo(**result.topics),
                tools_libraries=ToolsLibrariesInfo(**result.tools_libraries),
                summary=result.summary,
                key_strengths=result.key_strengths
            )
            
            logger.info("Resume parsing completed successfully")
            return resume_data
            
        except Exception as e:
            logger.error(f"Failed to parse resume: {str(e)}")
            raise ResumeMatcherException(f"Resume parsing failed: {str(e)}")
    
    async def parse_job_description(self, job_text: str) -> JobDescription:
        """
        Parse job description using LangChain agent.
        
        Args:
            job_text: Raw job description text
            
        Returns:
            JobDescription object with structured requirements
        """
        try:
            logger.info("Starting job description parsing with LangChain agent")
            
            # Create chain
            chain = self.job_prompt | self.llm | self.job_parser
            
            # Parse job description
            result = await chain.ainvoke({
                "job_text": job_text,
                "format_instructions": self.job_parser.get_format_instructions()
            })
            
            # Convert to JobDescription object
            job_data = JobDescription(
                title=result.title,
                company=result.company,
                raw_text=job_text,
                required_skills=result.required_skills,
                preferred_skills=result.preferred_skills,
                experience_years=result.experience_years,
                education_level=result.education_level,
                responsibilities=result.responsibilities
            )
            
            logger.info("Job description parsing completed successfully")
            return job_data
            
        except Exception as e:
            logger.error(f"Failed to parse job description: {str(e)}")
            raise ResumeMatcherException(f"Job description parsing failed: {str(e)}")
    
    async def analyze_match(self, resume_data: ResumeData, job_data: JobDescription) -> MatchResult:
        """
        Analyze how well a candidate matches a job using LangChain agent.
        
        Args:
            resume_data: Structured resume data
            job_data: Structured job description data
            
        Returns:
            MatchResult with detailed analysis and scores
        """
        try:
            logger.info(f"Starting match analysis for resume {resume_data.id} and job {job_data.id}")
            
            # Prepare candidate and job summaries for analysis
            candidate_summary = {
                "profile": resume_data.profile.__dict__,
                "experience_years": resume_data.experience.total_years,
                "skills": resume_data.skills.__dict__,
                "key_strengths": resume_data.key_strengths,
                "summary": resume_data.summary
            }
            
            job_summary = {
                "title": job_data.title,
                "required_skills": job_data.required_skills,
                "preferred_skills": job_data.preferred_skills,
                "experience_years": job_data.experience_years,
                "responsibilities": job_data.responsibilities
            }
            
            # Create chain
            chain = self.match_prompt | self.llm | self.match_parser
            
            # Analyze match
            result = await chain.ainvoke({
                "candidate_data": str(candidate_summary),
                "job_data": str(job_summary),
                "format_instructions": self.match_parser.get_format_instructions()
            })
            
            # Convert to MatchResult object
            match_result = MatchResult(
                resume_id=resume_data.id,
                job_id=job_data.id,
                overall_score=result.overall_score,
                skills_match_score=result.skills_match_score,
                experience_match_score=result.experience_match_score,
                semantic_similarity_score=0.0,  # Will be set by vector similarity
                matching_skills=result.matching_skills,
                missing_skills=result.missing_skills,
                strength_areas=result.strength_areas,
                improvement_areas=result.improvement_areas,
                match_summary=result.match_summary,
                recommendation=result.recommendation
            )
            
            logger.info("Match analysis completed successfully")
            return match_result
            
        except Exception as e:
            logger.error(f"Failed to analyze match: {str(e)}")
            raise ResumeMatcherException(f"Match analysis failed: {str(e)}")
    
    async def generate_summary(self, resume_data: ResumeData) -> str:
        """
        Generate an enhanced professional summary using LangChain.
        
        Args:
            resume_data: Structured resume data
            
        Returns:
            Enhanced professional summary
        """
        try:
            # Use the prompt from prompt manager
            chain = self.summary_prompt | self.llm
            
            result = await chain.ainvoke({
                "profile": f"{resume_data.profile.name} - {resume_data.profile.title}",
                "experience_years": resume_data.experience.total_years,
                "key_skills": ", ".join(resume_data.skills.technical[:5]),
                "strengths": ", ".join(resume_data.key_strengths)
            })
            
            return str(result.content).strip() if hasattr(result, 'content') else str(result).strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return resume_data.summary  # Fallback to original summary


# Global instance
langchain_agents = LangChainAgents()
