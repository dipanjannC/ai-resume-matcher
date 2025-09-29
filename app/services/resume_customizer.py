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
                "job_title": getattr(job_description, 'title', '') or "N/A",
                "company": getattr(job_description, 'company', '') or "N/A",
                "required_skills": ", ".join(getattr(job_description, 'required_skills', []) or []),
                "experience_required": getattr(job_description, 'experience_years', 0) or 0
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
                if hasattr(resume_data.skills, 'technical') and resume_data.skills.technical:
                    all_skills.extend(resume_data.skills.technical)
                if hasattr(resume_data.skills, 'soft') and resume_data.skills.soft:
                    all_skills.extend(resume_data.skills.soft)
            
            context = {
                "candidate_name": getattr(resume_data.profile, 'name', '') or "Dear Hiring Manager",
                "candidate_experience": getattr(resume_data, 'summary', '') or "",
                "candidate_skills": ", ".join(all_skills) if all_skills else "",
                "candidate_experience_years": getattr(resume_data.experience, 'total_years', 0) or 0,
                "job_title": getattr(job_description, 'title', '') or "",
                "company": getattr(job_description, 'company', '') or "",
                "job_requirements": (getattr(job_description, 'raw_text', '') or "")[:1000],
                "required_skills": ", ".join(getattr(job_description, 'required_skills', []) or []),
                "location": getattr(job_description, 'location', '') or "Remote"
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
                if hasattr(resume_data.skills, 'technical') and resume_data.skills.technical:
                    all_skills.extend(resume_data.skills.technical)
                if hasattr(resume_data.skills, 'soft') and resume_data.skills.soft:
                    all_skills.extend(resume_data.skills.soft)
            
            context = {
                "resume_skills": ", ".join(all_skills) if all_skills else "",
                "resume_experience": getattr(resume_data.experience, 'total_years', 0) or 0,
                "resume_summary": getattr(resume_data, 'summary', '') or "",
                "job_requirements": ", ".join(getattr(job_description, 'required_skills', []) or []),
                "job_experience_required": getattr(job_description, 'experience_years', 0) or 0,
                "job_description": (getattr(job_description, 'raw_text', '') or "")[:800]
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
            if hasattr(resume_data.skills, 'technical') and resume_data.skills.technical:
                all_skills.extend(resume_data.skills.technical)
            if hasattr(resume_data.skills, 'soft') and resume_data.skills.soft:
                all_skills.extend(resume_data.skills.soft)
        
        return f"""
Name: {getattr(resume_data.profile, 'name', '') or 'N/A'}
Email: {getattr(resume_data.profile, 'email', '') or 'N/A'}
Phone: {getattr(resume_data.profile, 'phone', '') or 'N/A'}
Title: {getattr(resume_data.profile, 'title', '') or 'N/A'}
Location: {getattr(resume_data.profile, 'location', '') or 'N/A'}
Summary: {getattr(resume_data, 'summary', '') or 'N/A'}
Experience Years: {getattr(resume_data.experience, 'total_years', 0) or 0}

Skills: {', '.join(all_skills) if all_skills else 'N/A'}
Technical Skills: {', '.join(getattr(resume_data.skills, 'technical', []) or []) if hasattr(resume_data.skills, 'technical') else 'N/A'}
Soft Skills: {', '.join(getattr(resume_data.skills, 'soft', []) or []) if hasattr(resume_data.skills, 'soft') else 'N/A'}
Certifications: {', '.join(getattr(resume_data.skills, 'certifications', []) or []) if hasattr(resume_data.skills, 'certifications') else 'N/A'}

Work Experience:
{self._format_experience_for_prompt(resume_data.experience) if resume_data.experience else 'N/A'}

Tools & Technologies:
{self._format_tools_for_prompt(resume_data.tools_libraries) if resume_data.tools_libraries else 'N/A'}

Key Strengths: {', '.join(getattr(resume_data, 'key_strengths', []) or []) if hasattr(resume_data, 'key_strengths') else 'N/A'}
        """.strip()
    
    def _format_job_for_prompt(self, job_description: JobDescription) -> str:
        """Format job description for LLM prompt"""
        return f"""
Job Title: {getattr(job_description, 'title', '') or 'N/A'}
Company: {getattr(job_description, 'company', '') or 'N/A'}
Location: {getattr(job_description, 'location', '') or 'N/A'}
Experience Required: {getattr(job_description, 'experience_years', 0) or 0} years
Required Skills: {', '.join(getattr(job_description, 'required_skills', []) or [])}
Job Summary: {getattr(job_description, 'summary', '') or 'N/A'}

Full Description:
{(getattr(job_description, 'raw_text', '') or 'N/A')[:1000]}...
        """.strip()
    
    def _format_experience_for_prompt(self, experience_info) -> str:
        """Format work experience for prompt"""
        if not experience_info:
            return "No work experience listed"
        
        formatted = []
        
        # Add total experience
        total_years = getattr(experience_info, 'total_years', 0)
        if total_years:
            formatted.append(f"Total Experience: {total_years} years")
        
        # Add roles and companies
        roles = getattr(experience_info, 'roles', []) or []
        if roles:
            formatted.append(f"Roles: {', '.join(roles)}")
        
        companies = getattr(experience_info, 'companies', []) or []
        if companies:
            formatted.append(f"Companies: {', '.join(companies)}")
        
        # Add key responsibilities
        responsibilities = getattr(experience_info, 'responsibilities', []) or []
        if responsibilities:
            formatted.append("Key Responsibilities:")
            for resp in responsibilities[:3]:  # Limit to top 3
                formatted.append(f"• {resp}")
        
        # Add achievements
        achievements = getattr(experience_info, 'achievements', []) or []
        if achievements:
            formatted.append("Key Achievements:")
            for achievement in achievements[:3]:  # Limit to top 3
                formatted.append(f"• {achievement}")
        
        return "\n".join(formatted) if formatted else "No work experience details available"
    
    def _format_tools_for_prompt(self, tools_libraries) -> str:
        """Format tools and technologies for prompt"""
        if not tools_libraries:
            return "No tools/technologies listed"
        
        formatted = []
        
        programming_languages = getattr(tools_libraries, 'programming_languages', []) or []
        if programming_languages:
            formatted.append(f"Programming Languages: {', '.join(programming_languages)}")
        
        frameworks = getattr(tools_libraries, 'frameworks', []) or []
        if frameworks:
            formatted.append(f"Frameworks: {', '.join(frameworks)}")
        
        tools = getattr(tools_libraries, 'tools', []) or []
        if tools:
            formatted.append(f"Tools: {', '.join(tools)}")
        
        databases = getattr(tools_libraries, 'databases', []) or []
        if databases:
            formatted.append(f"Databases: {', '.join(databases)}")
        
        if hasattr(tools_libraries, 'cloud_platforms') and tools_libraries.cloud_platforms:
            formatted.append(f"Cloud Platforms: {', '.join(tools_libraries.cloud_platforms)}")
        
        return "\n".join(formatted) if formatted else "No tools/technologies available"


# Create singleton instance
resume_customizer = ResumeCustomizerService()