"""
Enhanced LangChain agents for intelligent resume and job description processing.
Focused on clean, human-readable AI-powered parsing and analysis.
"""

import asyncio
import json
import re
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
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean LLM response to extract valid JSON.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
        
        # Find JSON content between braces
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        
        return response_text.strip()
    
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
            
            # Get format instructions
            format_instructions = self.resume_parser.get_format_instructions()
            
            # Create LLM chain without parser first
            llm_chain = self.resume_prompt | self.llm
            
            # Get raw LLM response
            raw_response = await llm_chain.ainvoke({
                "resume_text": resume_text,
                "format_instructions": format_instructions
            })
            
            logger.info(f"Raw LLM response: {raw_response.content[:500]}...")
            
            # Clean and parse the response
            try:
                cleaned_response = self._clean_json_response(str(raw_response.content))
                result = self.resume_parser.parse(cleaned_response)
            except Exception as parse_error:
                logger.error(f"Failed to parse LLM output: {str(parse_error)}")
                logger.error(f"Raw output was: {raw_response.content}")
                
                # Fallback: create a basic structure
                result = self._create_fallback_resume_structure(resume_text)
            
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
    
    def _create_fallback_resume_structure(self, resume_text: str) -> ResumeParsingOutput:
        """Create a basic fallback structure when parsing fails"""
        return ResumeParsingOutput(
            profile={
                "name": "Unknown",
                "title": "Unknown",
                "email": "",
                "phone": "",
                "linkedin": "",
                "location": ""
            },
            experience={
                "total_years": 0,
                "roles": [],
                "companies": [],
                "responsibilities": [],
                "achievements": []
            },
            skills={
                "technical": [],
                "soft": [],
                "certifications": [],
                "languages": []
            },
            topics={
                "domains": [],
                "specializations": [],
                "interests": []
            },
            tools_libraries={
                "programming_languages": [],
                "frameworks": [],
                "tools": [],
                "databases": [],
                "cloud_platforms": []
            },
            summary="Resume parsing failed - manual review required",
            key_strengths=["Manual review required"]
        )
    
    def _create_fallback_job_structure(self, job_text: str) -> JobParsingOutput:
        """Create a basic fallback structure when job parsing fails"""
        return JobParsingOutput(
            title="Unknown Position",
            company="Unknown Company",
            required_skills=[],
            preferred_skills=[],
            experience_years=0,
            education_level="Not specified",
            responsibilities=[],
            requirements=[],
            company_info="",
            summary="Job parsing failed - manual review required"
        )
    
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
            
            # Get format instructions
            format_instructions = self.job_parser.get_format_instructions()
            
            # Create LLM chain without parser first
            llm_chain = self.job_prompt | self.llm
            
            # Get raw LLM response
            raw_response = await llm_chain.ainvoke({
                "job_text": job_text,
                "format_instructions": format_instructions
            })
            
            logger.info(f"Raw job parsing response: {raw_response.content[:300]}...")
            
            # Clean and parse the response
            try:
                cleaned_response = self._clean_json_response(str(raw_response.content))
                result = self.job_parser.parse(cleaned_response)
            except Exception as parse_error:
                logger.error(f"Failed to parse job LLM output: {str(parse_error)}")
                logger.error(f"Raw output was: {raw_response.content}")
                
                # Fallback: create a basic structure
                result = self._create_fallback_job_structure(job_text)
            
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
                candidate_name=resume_data.profile.name or f"Candidate {resume_data.id[:8]}",
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
