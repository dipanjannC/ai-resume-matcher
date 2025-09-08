from langchain_core.prompts import ChatPromptTemplate


class LangChainPromptManager:
    """
    Centralized prompt manager for all LangChain agents.
    Manages prompts for resume parsing, job analysis, and matching.
    """

    def __init__(self) -> None:
        self._create_resume_parsing_prompts()
        self._create_job_parsing_prompts()
        self._create_matching_prompts()
        self._create_legacy_prompts()

    def _create_resume_parsing_prompts(self):
        """Create prompts for resume parsing agent"""
        self.resume_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI resume parser. Extract comprehensive structured information from resumes.
            
            Focus on:
            - Accurate profile information extraction
            - Detailed experience analysis with years calculation
            - Comprehensive skills categorization
            - Professional topics and domain expertise
            - Technical tools and technologies
            - Generate a compelling professional summary
            
            Be thorough but concise. Ensure data quality and accuracy."""),
            
            ("human", """Parse this resume and extract structured information:

{resume_text}

{format_instructions}""")
        ])

    def _create_job_parsing_prompts(self):
        """Create prompts for job description parsing agent"""
        self.job_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert job description analyzer. Extract key requirements and details from job postings.
            
            Focus on:
            - Clear identification of required vs preferred skills
            - Accurate experience requirements
            - Key responsibilities breakdown
            - Education requirements
            - Company and role identification
            
            Distinguish between must-have and nice-to-have requirements."""),
            
            ("human", """Analyze this job description and extract requirements:

{job_text}

{format_instructions}""")
        ])

    def _create_matching_prompts(self):
        """Create prompts for candidate-job matching analysis"""
        self.match_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert talent matching specialist. Analyze how well a candidate matches a job requirement.
            
            Provide:
            - Detailed scoring for skills and experience
            - Clear identification of matches and gaps
            - Actionable insights and recommendations
            - Professional assessment summary
            
            Be objective, fair, and constructive in your analysis."""),
            
            ("human", """Analyze this candidate-job match:

CANDIDATE PROFILE:
{candidate_data}

JOB REQUIREMENTS:
{job_data}

Provide detailed matching analysis with scores and recommendations.

{format_instructions}""")
        ])

        # Summary enhancement prompt
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional resume writer. Create compelling, concise professional summaries.
            
            Guidelines:
            - 2-3 sentences maximum
            - Highlight key strengths and experience
            - Focus on value proposition
            - Use active language
            - Be specific about expertise"""),
            
            ("human", """Create a professional summary for this candidate:

Profile: {profile}
Experience: {experience_years} years
Key Skills: {key_skills}
Strengths: {strengths}""")
        ])

    def _create_legacy_prompts(self):
        """Create legacy prompts for backward compatibility"""
        self.ROLE_PROMPT = """ 
You are an AI-powered Resume Matching Assistant. 
Your role is to analyze a candidate's resume in relation to a job description
and provide a structured, professional, and unbiased evaluation.  

You should:
  * Clearly restate the role requirements before comparing.
  * Identify the candidate's relevant skills, experience, and achievements.
  * Highlight strong matches between the resume and job description.
  * Point out any gaps or missing qualifications.
  * Provide an overall match score (0–100) with reasoning.
  * Remain objective, concise, and fair in your evaluation.
  * If the input is incomplete (e.g., missing resume or job description), ask for clarification.
"""

        self.RETRIEVER_PROMPT = """
You will be provided with retrieved context that may include resumes,
job descriptions, or skill requirements.  

When forming your evaluation:
  * First analyze the retrieved information and extract the most relevant details.
  * Explicitly reference them with phrases like "Based on the job description…" or 
    "From the candidate's resume…".
  * Connect how the retrieved details support or weaken the candidate–job alignment.

{context}

"""

        self.USER_PROMPT = """
[INST] 
Job Description: {job_description}  

Candidate Resume: {resume}  
[/INST]

Answer:
"""

    def get_resume_parsing_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for resume parsing"""
        return self.resume_prompt

    def get_job_parsing_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for job description parsing"""
        return self.job_prompt

    def get_matching_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for candidate-job matching"""
        return self.match_prompt

    def get_summary_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for summary enhancement"""
        return self.summary_prompt

    def get_chat_template(
        self, has_context: bool = False, has_history: bool = False
    ) -> ChatPromptTemplate:
        """
        Constructs a LangChain ChatPromptTemplate for resume–job matching.

        Args:
            has_context (bool): Whether retrieved context (e.g., skills DB) is available.
            has_history (bool): Whether chat history is available.

        Returns:
            ChatPromptTemplate: Configured chat prompt for evaluation.
        """
        messages = [
            ("system", self.ROLE_PROMPT),
        ]

        if has_context:
            messages.append(("system", self.RETRIEVER_PROMPT))

        if has_history:
            messages.append(("human", "{chat_history}"))

        messages.append(("user", self.USER_PROMPT))

        chat_prompt = ChatPromptTemplate.from_messages(messages)
        return chat_prompt

    def get_full_template(self) -> ChatPromptTemplate:
        """
        Returns a ChatPromptTemplate with all components included.

        Returns:
            ChatPromptTemplate: Complete template with optional components
        """
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.ROLE_PROMPT),
                ("system", self.RETRIEVER_PROMPT),  # Context is optional
                ("human", "{chat_history}"),        # Chat history is optional
                ("user", self.USER_PROMPT),
            ]
        )


# Legacy class for backward compatibility
class ResumeMatcherPrompt(LangChainPromptManager):
    """Legacy class for backward compatibility"""
    pass


# Global instance
prompt_manager = LangChainPromptManager()


if __name__ == "__main__":
    pass
