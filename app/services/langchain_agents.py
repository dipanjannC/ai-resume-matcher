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
                logger.debug(f"Cleaned response: {cleaned_response[:500]}...")
                result = self.resume_parser.parse(cleaned_response)
                logger.info("Successfully parsed resume with LangChain AI")
            except Exception as parse_error:
                logger.error(f"Failed to parse LLM output: {str(parse_error)}")
                logger.error(f"Raw output was: {str(raw_response.content)[:1000]}...")
                
                # Try multiple parsing strategies
                result = None
                
                # Strategy 1: Try to parse the raw response as JSON directly
                try:
                    import json
                    raw_content = str(raw_response.content)
                    raw_data = json.loads(raw_content)
                    logger.info("Raw response is valid JSON, attempting direct parsing")
                    
                    from app.models.langchain_models import ResumeParsingOutput
                    result = ResumeParsingOutput(**raw_data)
                    logger.info("Successfully created result from raw JSON")
                    
                except Exception as direct_parse_error:
                    logger.error(f"Direct JSON parsing failed: {str(direct_parse_error)}")
                    
                    # Strategy 2: Try to extract JSON from the response
                    try:
                        import re
                        import json
                        from app.models.langchain_models import ResumeParsingOutput
                        
                        raw_content = str(raw_response.content)
                        # Look for JSON-like structure in the response
                        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(0)
                            json_data = json.loads(json_str)
                            result = ResumeParsingOutput(**json_data)
                            logger.info("Successfully extracted and parsed JSON from response")
                        else:
                            raise Exception("No JSON structure found in response")
                            
                    except Exception as extraction_error:
                        logger.error(f"JSON extraction also failed: {str(extraction_error)}")
                        
                        # Strategy 3: Final fallback with enhanced extraction
                        logger.info("All parsing strategies failed, using enhanced fallback")
                        result = self._create_fallback_resume_structure(resume_text)
            
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
            except Exception as conversion_error:
                logger.error(f"Failed to convert result to ResumeData: {str(conversion_error)}")
                logger.error(f"Result object skills: {getattr(result, 'skills', 'No skills attr')}")
                logger.error(f"Result object type: {type(result)}")
                # Create fallback resume data if conversion fails
                logger.info("Creating fallback ResumeData structure")
                fallback_result = self._create_fallback_resume_structure(resume_text)
                resume_data = ResumeData()
                resume_data.raw_text = resume_text
                resume_data.profile = ProfileInfo(**fallback_result.profile)
                resume_data.experience = ExperienceInfo(**fallback_result.experience)
                resume_data.skills = SkillsInfo(**fallback_result.skills)
                resume_data.topics = TopicsInfo(**fallback_result.topics)
                resume_data.tools_libraries = ToolsLibrariesInfo(**fallback_result.tools_libraries)
                resume_data.summary = fallback_result.summary
                resume_data.key_strengths = fallback_result.key_strengths
            
            logger.info("Resume parsing completed successfully")
            return resume_data
            
        except Exception as e:
            logger.error(f"Failed to parse resume: {str(e)}")
            raise ResumeMatcherException(f"Resume parsing failed: {str(e)}")
    
    def _create_fallback_resume_structure(self, resume_text: str) -> ResumeParsingOutput:
        """Create a fallback structure using rule-based extraction when AI parsing fails"""
        logger.info("Creating fallback resume structure with rule-based extraction")
        
        # Use rule-based extraction methods
        name = self._extract_name_fallback(resume_text)
        title = self._extract_title_fallback(resume_text)
        email = self._extract_email_fallback(resume_text)
        phone = self._extract_phone_fallback(resume_text)
        linkedin = self._extract_linkedin_fallback(resume_text)
        technical_skills = self._extract_technical_skills_fallback(resume_text)
        experience_years = self._estimate_experience_years_fallback(resume_text)
        
        logger.info(f"Fallback extraction - Name: {name}, Title: {title}, Skills: {len(technical_skills)}")
        
        return ResumeParsingOutput(
            profile={
                "name": name,
                "title": title,
                "email": email,
                "phone": phone,
                "linkedin": linkedin,
                "location": ""
            },
            experience={
                "total_years": experience_years,
                "roles": [],
                "companies": [],
                "responsibilities": [],
                "achievements": []
            },
            skills={
                "technical": technical_skills,
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
                "programming_languages": self._categorize_programming_languages(technical_skills),
                "frameworks": self._categorize_frameworks(technical_skills),
                "tools": self._categorize_tools(technical_skills),
                "databases": self._categorize_databases(technical_skills),
                "cloud_platforms": self._categorize_cloud_platforms(technical_skills)
            },
            summary=f"Resume processed using rule-based extraction. Extracted {len(technical_skills)} skills.",
            key_strengths=["Rule-based parsing applied", "Manual review recommended for detailed information"]
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
    
    def _extract_name_fallback(self, text: str) -> str:
        """Extract name using rule-based patterns"""
        import re
        
        lines = text.split('\n')
        
        # Common name patterns
        name_patterns = [
            r'^([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # First line name
            r'Name[:]\s*([A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Name: format
            r'([A-Z][A-Z\s]+)',  # All caps name
        ]
        
        # Try first few lines for name
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            if len(line) > 50 or len(line) < 5:  # Skip very long or short lines
                continue
                
            for pattern in name_patterns:
                match = re.search(pattern, line)
                if match:
                    name = match.group(1).strip()
                    # Validate it looks like a name
                    if len(name.split()) >= 2 and not any(char.isdigit() for char in name):
                        return name
        
        return "Unknown"
    
    def _extract_title_fallback(self, text: str) -> str:
        """Extract title using rule-based patterns"""
        import re
        
        lines = text.split('\n')
        
        title_patterns = [
            r'(Engineer|Developer|Scientist|Manager|Analyst|Consultant|Specialist|Lead|Senior|Principal|Director|VP|President|CEO|CTO)',
            r'(Software|Data|Machine Learning|ML|AI|Full Stack|Backend|Frontend|DevOps|Cloud|System)',
        ]
        
        # Look in first 5 lines
        for line in lines[:5]:
            line = line.strip()
            if 2 < len(line) < 50:  # Reasonable title length
                for pattern in title_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        return line
        
        return "Unknown"
    
    def _extract_email_fallback(self, text: str) -> str:
        """Extract email using regex"""
        import re
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        match = re.search(email_pattern, text)
        return match.group(0) if match else ""
    
    def _extract_phone_fallback(self, text: str) -> str:
        """Extract phone using regex"""
        import re
        phone_patterns = [
            r'\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            r'\+?\d{10,15}',
        ]
        
        for pattern in phone_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return ""
    
    def _extract_linkedin_fallback(self, text: str) -> str:
        """Extract LinkedIn URL"""
        import re
        linkedin_pattern = r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+'
        match = re.search(linkedin_pattern, text, re.IGNORECASE)
        return match.group(0) if match else ""
    
    def _extract_technical_skills_fallback(self, text: str) -> list:
        """Extract technical skills using comprehensive keyword matching"""
        import re
        
        # Comprehensive skill lists
        programming_languages = [
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 'Go', 'Rust', 'Ruby', 'PHP',
            'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'SQL', 'HTML', 'CSS', 'Shell', 'Bash', 'PowerShell'
        ]
        
        frameworks_libraries = [
            'React', 'Vue', 'Angular', 'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 'Spring', 'Laravel',
            'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 'Pandas', 'NumPy', 'OpenCV', 'NLTK', 'SpaCy',
            'LangChain', 'Transformers', 'BERT', 'GPT', 'LLMs', 'Hugging Face'
        ]
        
        databases = [
            'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 'DynamoDB', 'Oracle',
            'SQL Server', 'SQLite', 'Neo4j', 'ChromaDB', 'Pinecone', 'Weaviate', 'Qdrant', 'Stardog'
        ]
        
        cloud_platforms = [
            'AWS', 'Azure', 'GCP', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'GitLab', 'GitHub Actions',
            'Terraform', 'Ansible', 'Chef', 'Puppet'
        ]
        
        tools_tech = [
            'Git', 'SVN', 'Jira', 'Confluence', 'Slack', 'Teams', 'Zoom', 'Figma', 'Adobe', 'Photoshop',
            'Linux', 'Unix', 'Windows', 'macOS', 'Apache', 'Nginx', 'Hadoop', 'Spark', 'Kafka', 'RabbitMQ',
            'Solr', 'Lucene', 'GraphQL', 'REST', 'API', 'Microservices', 'Serverless', 'Lambda'
        ]
        
        all_skills = programming_languages + frameworks_libraries + databases + cloud_platforms + tools_tech
        
        found_skills = []
        text_upper = text.upper()
        
        for skill in all_skills:
            # Use word boundaries for exact matches
            pattern = r'\b' + re.escape(skill.upper()) + r'\b'
            if re.search(pattern, text_upper):
                found_skills.append(skill)
        
        # Remove duplicates and sort
        found_skills = sorted(list(set(found_skills)))
        
        return found_skills
    
    def _estimate_experience_years_fallback(self, text: str) -> int:
        """Estimate experience years from text"""
        import re
        
        # Look for experience patterns
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'experience.*?(\d+)\+?\s*years?',
            r'(\d+)\+?\s*years?\s*in',
        ]
        
        max_years = 0
        for pattern in experience_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = int(match)
                    max_years = max(max_years, years)
                except ValueError:
                    continue
        
        return max_years
    
    def _categorize_programming_languages(self, skills: list) -> list:
        """Categorize programming languages from technical skills"""
        programming_langs = {
            'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 'Go', 'Rust', 
            'Ruby', 'PHP', 'Swift', 'Kotlin', 'Scala', 'R', 'MATLAB', 'SQL', 'HTML', 'CSS'
        }
        return [skill for skill in skills if skill in programming_langs]
    
    def _categorize_frameworks(self, skills: list) -> list:
        """Categorize frameworks from technical skills"""
        frameworks = {
            'React', 'Vue', 'Angular', 'Node.js', 'Express', 'Django', 'Flask', 'FastAPI', 
            'Spring', 'Laravel', 'TensorFlow', 'PyTorch', 'Keras', 'Scikit-learn', 
            'LangChain', 'Transformers', 'Hugging Face'
        }
        return [skill for skill in skills if skill in frameworks]
    
    def _categorize_tools(self, skills: list) -> list:
        """Categorize tools from technical skills"""
        tools = {
            'Git', 'Docker', 'Kubernetes', 'Jenkins', 'GitLab', 'GitHub Actions', 
            'Jira', 'Confluence', 'Tableau', 'PowerBI', 'Jupyter', 'VS Code',
            'Linux', 'Unix', 'Apache', 'Nginx'
        }
        return [skill for skill in skills if skill in tools]
    
    def _categorize_databases(self, skills: list) -> list:
        """Categorize databases from technical skills"""
        databases = {
            'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 
            'DynamoDB', 'Oracle', 'SQL Server', 'SQLite', 'Neo4j', 'ChromaDB', 
            'Pinecone', 'Weaviate', 'Qdrant', 'Stardog', 'Solr'
        }
        return [skill for skill in skills if skill in databases]
    
    def _categorize_cloud_platforms(self, skills: list) -> list:
        """Categorize cloud platforms from technical skills"""
        cloud_platforms = {
            'AWS', 'Azure', 'GCP', 'Google Cloud', 'Heroku', 'DigitalOcean', 
            'Vercel', 'Netlify', 'Firebase'
        }
        return [skill for skill in skills if skill in cloud_platforms]
    
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
            
            # Create chain
            chain = prompt | self.llm
            
            # Invoke chain with context
            result = await chain.ainvoke(context)
            
            # Parse JSON response
            response_text = str(result.content) if hasattr(result, 'content') else str(result)
            
            # Clean and parse JSON
            cleaned_response = self._clean_json_response(response_text)
            customization_data = json.loads(cleaned_response)
            
            return customization_data
            
        except Exception as e:
            logger.error(f"Failed to customize resume: {str(e)}")
            return None

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
