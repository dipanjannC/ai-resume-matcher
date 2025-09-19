#!/usr/bin/env python3
"""
Resume and Cover Letter Customization Service
Customizes resumes and generates cover letters based on job descriptions using LangChain agents
"""

import asyncio
from typing import Optional, Dict, Any, List
from pathlib import Path

from app.models.resume_data import ResumeData, JobDescription
from app.services.langchain_agents import langchain_agents
from app.services.prompt_manager import prompt_manager
from app.core.logging import get_logger

logger = get_logger(__name__)


class ResumeCustomizerService:
    """Service for customizing resumes and generating cover letters based on job descriptions"""
    
    def __init__(self):
        self.langchain_agents = langchain_agents
        self.prompt_manager = prompt_manager
        
    async def customize_resume(self, resume_data: ResumeData, job_description: JobDescription) -> Dict[str, Any]:
        """
        Customize a resume based on a specific job description
        
        Args:
            resume_data: Original resume data
            job_description: Target job description
            
        Returns:
            Dictionary containing customized resume sections
        """
        try:
            logger.info(f"Customizing resume for job: {job_description.title}")
            
            # Get customization prompt
            prompt = self.prompt_manager.get_resume_customization_prompt()
            
            # Prepare context for LLM
            context = {
                "original_resume": self._format_resume_for_prompt(resume_data),
                "job_description": self._format_job_for_prompt(job_description),
                "job_title": job_description.title or "N/A",
                "company": job_description.company or "N/A",
                "required_skills": ", ".join(job_description.required_skills) if job_description.required_skills else "",
                "experience_required": job_description.experience_years or 0
            }
            
            # Generate customized resume using LangChain
            customized_content = await self.langchain_agents.customize_resume_for_job(
                context, prompt
            )
            
            if customized_content:
                return {
                    "success": True,
                    "customized_resume": customized_content,
                    "job_title": job_description.title,
                    "company": job_description.company,
                    "customization_summary": customized_content.get("customization_summary", "")
                }
            else:
                return {"success": False, "error": "Failed to generate customized resume"}
                
        except Exception as e:
            logger.error(f"Error customizing resume: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def generate_cover_letter(self, resume_data: ResumeData, job_description: JobDescription) -> Dict[str, Any]:
        """
        Generate a personalized cover letter based on resume and job description
        
        Args:
            resume_data: Candidate's resume data
            job_description: Target job description
            
        Returns:
            Dictionary containing generated cover letter
        """
        try:
            logger.info(f"Generating cover letter for job: {job_description.title}")
            
            # Get cover letter prompt
            prompt = self.prompt_manager.get_cover_letter_prompt()
            
            # Prepare context for LLM
            all_skills = []
            if resume_data.skills:
                if resume_data.skills.technical:
                    all_skills.extend(resume_data.skills.technical)
                if resume_data.skills.soft:
                    all_skills.extend(resume_data.skills.soft)
            
            context = {
                "candidate_name": resume_data.profile.name or "Dear Hiring Manager",
                "candidate_experience": resume_data.summary or "",
                "candidate_skills": ", ".join(all_skills),
                "candidate_experience_years": resume_data.experience.total_years or 0,
                "job_title": job_description.title,
                "company": job_description.company,
                "job_requirements": job_description.raw_text[:1000] if job_description.raw_text else "",  # First 1000 chars
                "required_skills": ", ".join(job_description.required_skills) if job_description.required_skills else "",
                "location": job_description.location or "Remote"
            }
            
            # Generate cover letter using LangChain
            cover_letter = await self.langchain_agents.generate_cover_letter(
                context, prompt
            )
            
            if cover_letter:
                return {
                    "success": True,
                    "cover_letter": cover_letter,
                    "job_title": job_description.title,
                    "company": job_description.company,
                    "candidate_name": resume_data.profile.name or "Candidate"
                }
            else:
                return {"success": False, "error": "Failed to generate cover letter"}
                
        except Exception as e:
            logger.error(f"Error generating cover letter: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def get_customization_suggestions(self, resume_data: ResumeData, job_description: JobDescription) -> Dict[str, Any]:
        """
        Analyze resume against job description and provide customization suggestions
        
        Args:
            resume_data: Original resume data
            job_description: Target job description
            
        Returns:
            Dictionary containing customization suggestions
        """
        try:
            logger.info(f"Analyzing customization suggestions for job: {job_description.title}")
            
            # Get analysis prompt
            prompt = self.prompt_manager.get_customization_analysis_prompt()
            
            # Prepare context for analysis
            all_skills = []
            if resume_data.skills:
                all_skills.extend(resume_data.skills.technical or [])
                all_skills.extend(resume_data.skills.soft or [])
            
            context = {
                "resume_skills": ", ".join(all_skills),
                "resume_experience": resume_data.experience.total_years or 0,
                "resume_summary": resume_data.summary or "",
                "job_requirements": ", ".join(job_description.required_skills) if job_description.required_skills else "",
                "job_experience_required": job_description.experience_years or 0,
                "job_description": job_description.raw_text[:800] if job_description.raw_text else ""  # First 800 chars
            }
            
            # Generate suggestions using LangChain
            suggestions = await self.langchain_agents.analyze_customization_needs(
                context, prompt
            )
            
            if suggestions:
                return {
                    "success": True,
                    "suggestions": suggestions,
                    "skill_gaps": suggestions.get("skill_gaps", []),
                    "experience_recommendations": suggestions.get("experience_recommendations", []),
                    "keyword_suggestions": suggestions.get("keyword_suggestions", []),
                    "priority_changes": suggestions.get("priority_changes", [])
                }
            else:
                return {"success": False, "error": "Failed to generate suggestions"}
                
        except Exception as e:
            logger.error(f"Error analyzing customization suggestions: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _format_resume_for_prompt(self, resume_data: ResumeData) -> str:
        """Format resume data for LLM prompt"""
        all_skills = []
        if resume_data.skills:
            all_skills.extend(resume_data.skills.technical or [])
            all_skills.extend(resume_data.skills.soft or [])
        
        return f"""
Name: {resume_data.profile.name or 'N/A'}
Email: {resume_data.profile.email or 'N/A'}
Phone: {resume_data.profile.phone or 'N/A'}
Title: {resume_data.profile.title or 'N/A'}
Location: {resume_data.profile.location or 'N/A'}
Summary: {resume_data.summary or 'N/A'}
Experience Years: {resume_data.experience.total_years or 0}

Skills: {', '.join(all_skills) if all_skills else 'N/A'}
Technical Skills: {', '.join(resume_data.skills.technical) if resume_data.skills.technical else 'N/A'}
Soft Skills: {', '.join(resume_data.skills.soft) if resume_data.skills.soft else 'N/A'}
Certifications: {', '.join(resume_data.skills.certifications) if resume_data.skills.certifications else 'N/A'}

Work Experience:
{self._format_experience_for_prompt(resume_data.experience)}

Tools & Technologies:
{self._format_tools_for_prompt(resume_data.tools_libraries)}

Key Strengths: {', '.join(resume_data.key_strengths) if resume_data.key_strengths else 'N/A'}
        """.strip()
    
    def _format_job_for_prompt(self, job_description: JobDescription) -> str:
        """Format job description for LLM prompt"""
        return f"""
Job Title: {job_description.title or 'N/A'}
Company: {job_description.company or 'N/A'}
Location: {job_description.location or 'N/A'}
Experience Required: {job_description.experience_years or 0} years
Required Skills: {', '.join(job_description.required_skills) if job_description.required_skills else 'N/A'}
Job Summary: {job_description.summary or 'N/A'}

Full Description:
{job_description.raw_text[:1000] if job_description.raw_text else 'N/A'}...
        """.strip()
    
    def _format_experience_for_prompt(self, experience_info) -> str:
        """Format work experience for prompt"""
        if not experience_info or not hasattr(experience_info, 'roles'):
            return "No work experience listed"
        
        formatted = []
        
        # Add total experience
        if experience_info.total_years:
            formatted.append(f"Total Experience: {experience_info.total_years} years")
        
        # Add roles and companies
        if experience_info.roles:
            formatted.append(f"Roles: {', '.join(experience_info.roles)}")
        
        if experience_info.companies:
            formatted.append(f"Companies: {', '.join(experience_info.companies)}")
        
        # Add key responsibilities
        if experience_info.responsibilities:
            formatted.append("Key Responsibilities:")
            for resp in experience_info.responsibilities[:3]:  # Limit to top 3
                formatted.append(f"• {resp}")
        
        # Add achievements
        if experience_info.achievements:
            formatted.append("Key Achievements:")
            for achievement in experience_info.achievements[:3]:  # Limit to top 3
                formatted.append(f"• {achievement}")
        
        return "\n".join(formatted) if formatted else "No work experience details available"
    
    def _format_tools_for_prompt(self, tools_libraries) -> str:
        """Format tools and technologies for prompt"""
        if not tools_libraries:
            return "No tools/technologies listed"
        
        formatted = []
        
        if hasattr(tools_libraries, 'programming_languages') and tools_libraries.programming_languages:
            formatted.append(f"Programming Languages: {', '.join(tools_libraries.programming_languages)}")
        
        if hasattr(tools_libraries, 'frameworks') and tools_libraries.frameworks:
            formatted.append(f"Frameworks: {', '.join(tools_libraries.frameworks)}")
        
        if hasattr(tools_libraries, 'tools') and tools_libraries.tools:
            formatted.append(f"Tools: {', '.join(tools_libraries.tools)}")
        
        if hasattr(tools_libraries, 'databases') and tools_libraries.databases:
            formatted.append(f"Databases: {', '.join(tools_libraries.databases)}")
        
        if hasattr(tools_libraries, 'cloud_platforms') and tools_libraries.cloud_platforms:
            formatted.append(f"Cloud Platforms: {', '.join(tools_libraries.cloud_platforms)}")
        
        return "\n".join(formatted) if formatted else "No tools/technologies available"


# Create singleton instance
resume_customizer = ResumeCustomizerService()