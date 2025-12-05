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
        """Initialize LangChain agents with multiple LLM providers for fallback"""
        self.llms = []
        
        # 1. Primary: Groq
        try:
            self.llms.append(("Groq", LLMService().get_groq()))
        except Exception as e:
            logger.warning(f"Groq LLM not available: {e}")
            
        # 2. Secondary: Gemini
        try:
            self.llms.append(("Gemini", LLMService().get_gemini()))
        except Exception as e:
            logger.warning(f"Gemini LLM not available: {e}")
            
        # 3. Tertiary: OpenAI (if key provided or in env)
        try:
            # Check if OpenAI key is available
            if openai_api_key or os.getenv("OPENAI_API_KEY"):
                from langchain_openai import ChatOpenAI
                self.llms.append(("OpenAI", ChatOpenAI(
                    model="gpt-3.5-turbo",
                    temperature=0.3,
                    api_key=openai_api_key or os.getenv("OPENAI_API_KEY")
                )))
        except Exception as e:
            logger.warning(f"OpenAI LLM not available: {e}")
            
        if not self.llms:
            logger.error("No LLM providers available! Please configure at least one provider.")
            # We don't raise here to allow app to start, but operations will fail
        
        # Initialize parsers
        self.resume_parser = PydanticOutputParser(pydantic_object=ResumeParsingOutput)
        self.job_parser = PydanticOutputParser(pydantic_object=JobParsingOutput)
        self.match_parser = PydanticOutputParser(pydantic_object=MatchAnalysisOutput)
        
        # Get prompt templates from prompt manager
        self.resume_prompt = prompt_manager.get_resume_parsing_prompt()
        self.job_prompt = prompt_manager.get_job_parsing_prompt()
        self.match_prompt = prompt_manager.get_matching_prompt()
        self.summary_prompt = prompt_manager.get_summary_prompt()
        
        logger.info(f"LangChain agents initialized with {len(self.llms)} providers: {[name for name, _ in self.llms]}")

    async def _execute_with_fallback(self, operation_name: str, prompt_template: ChatPromptTemplate, input_data: Dict[str, Any]) -> Any:
        """
        Execute an LLM operation with automatic fallback to available providers.
        
        Args:
            operation_name: Name of operation for logging
            prompt_template: LangChain prompt template
            input_data: Input dictionary for the prompt
            
        Returns:
            LLM response content
            
        Raises:
            ResumeMatcherException: If all providers fail
        """
        last_error = None
        
        for provider_name, llm in self.llms:
            try:
                logger.info(f"Attempting {operation_name} with {provider_name}...")
                chain = prompt_template | llm
                response = await chain.ainvoke(input_data)
                logger.info(f"{operation_name} successful with {provider_name}")
                return response
            except Exception as e:
                logger.warning(f"{operation_name} failed with {provider_name}: {e}")
                last_error = e
                continue
                
        # If we get here, all providers failed
        error_msg = f"All LLM providers failed for {operation_name}. Last error: {last_error}"
        logger.error(error_msg)
        raise ResumeMatcherException(error_msg)

    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean LLM response to extract valid JSON and handle null values.
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Cleaned JSON string with null values replaced
        """
        logger.debug(f"Cleaning JSON response: {response_text[:200]}...")
        
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```json\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'^```\s*$', '', response_text, flags=re.MULTILINE)
        response_text = response_text.strip()
        
        # Find JSON content between braces - handle nested braces
        brace_count = 0
        start_idx = -1
        end_idx = -1
        
        for i, char in enumerate(response_text):
            if char == '{':
                if start_idx == -1:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and start_idx != -1:
                    end_idx = i
                    break
        
        if start_idx != -1 and end_idx != -1:
            json_text = response_text[start_idx:end_idx + 1]
        else:
            json_text = response_text
        
        # Parse and clean the JSON to handle null values
        try:
            parsed_json = json.loads(json_text)
            
            # Replace null values and fix type mismatches
            def clean_nulls(obj):
                if isinstance(obj, dict):
                    for key, value in obj.items():
                        if value is None:
                            # Set appropriate defaults based on expected types
                            if key in ['title', 'company', 'education_level', 'company_info', 'summary', 'name', 'email', 'phone', 'linkedin', 'location']:
                                obj[key] = ""
                            elif key in ['required_skills', 'preferred_skills', 'responsibilities', 'requirements', 'technical', 'soft', 'certifications', 'languages', 'roles', 'companies', 'achievements', 'domains', 'specializations', 'interests', 'programming_languages', 'frameworks', 'tools', 'databases', 'cloud_platforms', 'key_strengths']:
                                obj[key] = []
                            elif key in ['experience_years', 'total_years']:
                                obj[key] = 0
                        # Fix type mismatches for expected dictionary fields
                        elif key in ['topics', 'tools_libraries', 'skills', 'experience', 'profile'] and isinstance(value, list) and len(value) == 0:
                            # Convert empty list to empty dict for dictionary fields
                            obj[key] = {}
                        elif key == 'topics' and isinstance(value, list):
                            # Convert list to proper topics structure
                            obj[key] = {
                                'domains': value if isinstance(value, list) else [],
                                'specializations': [],
                                'interests': []
                            }
                        else:
                            obj[key] = clean_nulls(value)
                elif isinstance(obj, list):
                    return [clean_nulls(item) for item in obj if item is not None]
                return obj
            
            cleaned_json = clean_nulls(parsed_json)
            result = json.dumps(cleaned_json)
            logger.debug(f"JSON cleaned successfully: {len(result)} characters")
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.error(f"Attempted to parse: {json_text[:500]}...")
            # If JSON parsing fails, return original text
            return json_text
    
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
            
            # Execute with fallback
            raw_response = await self._execute_with_fallback(
                "Resume Parsing",
                self.resume_prompt,
                {
                    "resume_text": resume_text,
                    "format_instructions": format_instructions
                }
            )
            
            logger.info(f"Raw LLM response: {str(raw_response.content)[:500]}...")
            
            # Clean and parse the response
            try:
                cleaned_response = self._clean_json_response(str(raw_response.content))
                logger.debug(f"Cleaned response: {cleaned_response[:500]}...")
                result = self.resume_parser.parse(cleaned_response)
                logger.info("Successfully parsed resume with LangChain AI")
            except Exception as parse_error:
                logger.error(f"Failed to parse LLM output: {str(parse_error)}")
                logger.error(f"Raw output was: {str(raw_response.content)[:1000]}...")
                
                # Try to parse raw JSON if parser failed
                try:
                    import json
                    raw_content = str(raw_response.content)
                    # Try to find JSON block
                    import re
                    json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                    if json_match:
                        json_data = json.loads(json_match.group(0))
                        from app.models.langchain_models import ResumeParsingOutput
                        result = ResumeParsingOutput(**json_data)
                    else:
                        raise ValueError("No JSON found")
                except Exception as e:
                    logger.error(f"All parsing attempts failed: {e}")
                    raise ResumeMatcherException("Failed to parse resume data from AI response. Please try again.")

            # Convert to ResumeData object
            try:
                # Validate and fix the result structure before conversion
                result = self._validate_and_fix_result_structure(result)
                
                resume_data = ResumeData()
                resume_data.raw_text = resume_text
                resume_data.profile = ProfileInfo(**result.profile)
                resume_data.experience = ExperienceInfo(**result.experience)
                resume_data.skills = SkillsInfo(**result.skills)
                resume_data.topics = TopicsInfo(**result.topics)
                resume_data.tools_libraries = ToolsLibrariesInfo(**result.tools_libraries)
                resume_data.summary = result.summary
                resume_data.key_strengths = result.key_strengths
                
                logger.info("Resume parsing completed successfully")
                return resume_data
                
            except Exception as conversion_error:
                logger.error(f"Failed to convert result to ResumeData: {str(conversion_error)}")
                raise ResumeMatcherException("Failed to structure resume data. Please try again.")
            
        except Exception as e:
            logger.error(f"Failed to parse resume: {str(e)}")
            raise ResumeMatcherException(f"Resume parsing failed: {str(e)}")
    
    # Fallback methods removed in favor of multi-model rerouting
    
    def _validate_and_fix_result_structure(self, result):
        """Validate and fix the result structure to match expected models"""
        # Fix skills structure if it has unexpected keys
        if hasattr(result, 'skills') and isinstance(result.skills, dict):
            expected_skills_keys = {'technical', 'soft', 'certifications', 'languages'}
            actual_keys = set(result.skills.keys())
            
            if not actual_keys.issubset(expected_skills_keys):
                logger.warning(f"Skills has unexpected keys: {actual_keys - expected_skills_keys}")
                # Create a new skills dict with only expected keys
                new_skills = {
                    'technical': result.skills.get('technical', []),
                    'soft': result.skills.get('soft', []),
                    'certifications': result.skills.get('certifications', []),
                    'languages': result.skills.get('languages', [])
                }
                
                # If there are unexpected keys, try to merge them into technical
                for key, value in result.skills.items():
                    if key not in expected_skills_keys and isinstance(value, list):
                        new_skills['technical'].extend(value)
                        logger.info(f"Merged unexpected skills key '{key}' into technical skills")
                
                result.skills = new_skills
                
        # Fix tools_libraries structure if it has unexpected keys
        if hasattr(result, 'tools_libraries') and isinstance(result.tools_libraries, dict):
            expected_tools_keys = {'programming_languages', 'frameworks', 'tools', 'databases', 'cloud_platforms'}
            actual_keys = set(result.tools_libraries.keys())
            
            if not actual_keys.issubset(expected_tools_keys):
                logger.warning(f"Tools_libraries has unexpected keys: {actual_keys - expected_tools_keys}")
                # Create a new tools_libraries dict with only expected keys
                new_tools = {
                    'programming_languages': result.tools_libraries.get('programming_languages', []),
                    'frameworks': result.tools_libraries.get('frameworks', []),
                    'tools': result.tools_libraries.get('tools', []),
                    'databases': result.tools_libraries.get('databases', []),
                    'cloud_platforms': result.tools_libraries.get('cloud_platforms', [])
                }
                
                # Handle common variations and map them to correct fields
                key_mappings = {
                    'Programming_Languages': 'programming_languages',
                    'programming_language': 'programming_languages',  
                    'languages': 'programming_languages',
                    'framework': 'frameworks',
                    'tool': 'tools',
                    'database': 'databases',
                    'cloud_platform': 'cloud_platforms',
                    'cloud': 'cloud_platforms',
                    'Cloud_Platforms': 'cloud_platforms',
                    'Frameworks': 'frameworks',
                    'Tools': 'tools',
                    'Databases': 'databases'
                }
                
                # Map unexpected keys to correct fields
                for key, value in result.tools_libraries.items():
                    if key not in expected_tools_keys and isinstance(value, list):
                        mapped_key = key_mappings.get(key, 'tools')  # Default to 'tools' if no mapping
                        new_tools[mapped_key].extend(value)
                        logger.info(f"Mapped unexpected tools key '{key}' to '{mapped_key}'")
                
                result.tools_libraries = new_tools
                
        # Fix topics structure if needed
        if hasattr(result, 'topics') and isinstance(result.topics, dict):
            expected_topics_keys = {'domains', 'specializations', 'interests'}
            if not set(result.topics.keys()).issubset(expected_topics_keys):
                result.topics = {
                    'domains': result.topics.get('domains', []),
                    'specializations': result.topics.get('specializations', []),
                    'interests': result.topics.get('interests', [])
                }
        
        # Fix profile structure if it has unexpected keys
        if hasattr(result, 'profile') and isinstance(result.profile, dict):
            expected_profile_keys = {'name', 'title', 'email', 'phone', 'linkedin', 'location'}
            actual_keys = set(result.profile.keys())
            
            if not actual_keys.issubset(expected_profile_keys):
                logger.warning(f"Profile has unexpected keys: {actual_keys - expected_profile_keys}")
                # Create a new profile dict with only expected keys
                new_profile = {
                    'name': result.profile.get('name', ''),
                    'title': result.profile.get('title', ''),
                    'email': result.profile.get('email', ''),
                    'phone': result.profile.get('phone', ''),
                    'linkedin': result.profile.get('linkedin', ''),
                    'location': result.profile.get('location', '')
                }
                
                # Handle common profile field variations
                profile_key_mappings = {
                    'contact_info': 'phone',  # Map contact_info to phone
                    'contact': 'phone',
                    'telephone': 'phone',
                    'linkedin_url': 'linkedin',
                    'linkedin_profile': 'linkedin',
                    'address': 'location',
                    'city': 'location',
                    'job_title': 'title',
                    'position': 'title',
                    'role': 'title',
                    'full_name': 'name'
                }
                
                # Map unexpected keys to correct fields
                for key, value in result.profile.items():
                    if key not in expected_profile_keys:
                        mapped_key = profile_key_mappings.get(key)
                        if mapped_key and isinstance(value, str):
                            if not new_profile[mapped_key]:  # Only use if the field is empty
                                new_profile[mapped_key] = value
                                logger.info(f"Mapped unexpected profile key '{key}' to '{mapped_key}'")
                        else:
                            logger.info(f"Ignored unexpected profile key '{key}' with value: {value}")
                
                result.profile = new_profile
                
        return result
    
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
            
            # Execute with fallback
            raw_response = await self._execute_with_fallback(
                "Job Parsing",
                self.job_prompt,
                {
                    "job_text": job_text,
                    "format_instructions": format_instructions
                }
            )
            
            logger.info(f"Raw job parsing response: {str(raw_response.content)[:300]}...")
            
            # Clean and parse the response
            try:
                cleaned_response = self._clean_json_response(str(raw_response.content))
                result = self.job_parser.parse(cleaned_response)
            except Exception as parse_error:
                logger.error(f"Failed to parse job LLM output: {str(parse_error)}")
                logger.error(f"Raw output was: {raw_response.content}")
                raise ResumeMatcherException("Failed to parse job description from AI response. Please try again.")
            
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
            
            # Execute with fallback
            # Note: match_parser is part of the chain in original code, but _execute_with_fallback returns raw response
            # We need to parse it manually or include parser in the chain passed to _execute_with_fallback
            # But _execute_with_fallback takes a prompt_template and adds | llm.
            # So we get raw response and then parse.
            
            raw_response = await self._execute_with_fallback(
                "Match Analysis",
                self.match_prompt,
                {
                    "candidate_data": str(candidate_summary),
                    "job_data": str(job_summary),
                    "format_instructions": self.match_parser.get_format_instructions()
                }
            )
            
            # Parse response
            try:
                cleaned_response = self._clean_json_response(str(raw_response.content))
                result = self.match_parser.parse(cleaned_response)
            except Exception as parse_error:
                logger.error(f"Failed to parse match analysis output: {parse_error}")
                raise ResumeMatcherException("Failed to analyze match. Please try again.")
            
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
            # Execute with fallback
            result = await self._execute_with_fallback(
                "Summary Generation",
                self.summary_prompt,
                {
                    "profile": f"{resume_data.profile.name} - {resume_data.profile.title}",
                    "experience_years": resume_data.experience.total_years,
                    "key_skills": ", ".join(resume_data.skills.technical[:5]),
                    "strengths": ", ".join(resume_data.key_strengths)
                }
            )
            
            return str(result.content).strip() if hasattr(result, 'content') else str(result).strip()
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {str(e)}")
            return resume_data.summary  # Fallback to original summary

    async def customize_resume_for_job(self, context: Dict[str, Any], prompt: ChatPromptTemplate) -> Optional[Dict[str, Any]]:
        """
        Customize a resume for a specific job using LangChain
        
        Args:
            context: Dictionary containing resume and job information
            prompt: ChatPromptTemplate for customization
            
        Returns:
            Dictionary containing customized resume sections
        """
        try:
            logger.info("Customizing resume using LangChain agents")
            
            # Execute with fallback
            raw_response = await self._execute_with_fallback(
                "Resume Customization",
                prompt,
                context
            )
            
            # Clean and parse response
            cleaned_response = self._clean_json_response(str(raw_response.content))
            return json.loads(cleaned_response)
            
        except Exception as e:
            logger.error(f"Failed to customize resume: {e}")
            raise ResumeMatcherException(f"Resume customization failed: {str(e)}")

    async def generate_cover_letter(self, context: Dict[str, Any], prompt: ChatPromptTemplate) -> Optional[str]:
        """
        Generate a cover letter using LangChain
        
        Args:
            context: Dictionary containing candidate and job information
            prompt: ChatPromptTemplate for cover letter generation
            
        Returns:
            Generated cover letter text
        """
        try:
            logger.info("Generating cover letter using LangChain agents")
            
            # Create chain
            chain = prompt | self.llm
            
            # Invoke chain with context
            result = await chain.ainvoke(context)
            
            # Extract cover letter text
            cover_letter = str(result.content) if hasattr(result, 'content') else str(result)
            
            return cover_letter.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate cover letter: {str(e)}")
            return None

    async def analyze_customization_needs(self, context: Dict[str, Any], prompt: ChatPromptTemplate) -> Optional[Dict[str, Any]]:
        """
        Analyze customization needs for a resume using LangChain
        
        Args:
            context: Dictionary containing resume and job analysis information
            prompt: ChatPromptTemplate for customization analysis
            
        Returns:
            Dictionary containing customization suggestions
        """
        try:
            logger.info("Analyzing customization needs using LangChain agents")
            
            # Create chain
            chain = prompt | self.llm
            
            # Invoke chain with context
            result = await chain.ainvoke(context)
            
            # Parse JSON response
            response_text = str(result.content) if hasattr(result, 'content') else str(result)
            
            # Clean and parse JSON
            cleaned_response = self._clean_json_response(response_text)
            suggestions_data = json.loads(cleaned_response)
            
            return suggestions_data
            
        except Exception as e:
            logger.error(f"Failed to analyze customization needs: {str(e)}")
            return None


# Global instance
langchain_agents = LangChainAgents()


if __name__ == "__main__":
    # Simple test
    async def test():
        sample_resume = """
        John Doe
        Software Engineer with 5 years of experience in Python, Java, and cloud technologies.
        Skilled in developing scalable web applications and working in agile teams.
        """
        
        sample_job = """
        We are looking for a Software Engineer with at least 3 years of experience in Python and cloud platforms.
        Responsibilities include developing web applications and collaborating with cross-functional teams.
        """
        
        resume_data = await langchain_agents.parse_resume(sample_resume)
        job_data = await langchain_agents.parse_job_description(sample_job)
        match_result = await langchain_agents.analyze_match(resume_data, job_data)
        
        print("Match Result:", match_result)
    
    asyncio.run(test())
