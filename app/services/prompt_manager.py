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
        self._create_customization_prompts()
        self._create_legacy_prompts()

    def _create_resume_parsing_prompts(self):
        """Create prompts for resume parsing agent"""
        self.resume_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert AI resume parser. Extract comprehensive structured information from resumes.
            
            CRITICAL REQUIREMENTS:
            - You MUST respond with valid JSON format only
            - Do NOT include any explanatory text before or after the JSON
            - Use empty strings "" instead of null for missing text fields
            - Use empty arrays [] instead of null for missing list fields
            - Use 0 instead of null for missing numeric fields
            - Extract ALL available information accurately
            
            Focus on:
            - Accurate profile information extraction (name, title, contact info)
            - Detailed experience analysis with years calculation
            - Comprehensive skills categorization (technical, soft, certifications)
            - Professional topics and domain expertise
            - Technical tools and technologies
            - Generate a compelling professional summary
            
            Guidelines:
            - If information is not available, use appropriate empty defaults
            - Calculate total experience years from work history
            - Categorize skills appropriately
            - Be thorough but concise. Ensure data quality and accuracy."""),
            
            ("human", """Parse this resume and extract structured information:

{resume_text}

{format_instructions}

Return ONLY valid JSON format with no additional text. Do NOT use null values.""")
        ])

    def _create_job_parsing_prompts(self):
        """Create prompts for job description parsing agent"""
        self.job_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert job description analyzer. Extract key requirements and details from job postings.
            
            CRITICAL REQUIREMENTS:
            - You MUST respond with valid JSON format only
            - Do NOT include any explanatory text before or after the JSON
            - Use empty strings "" instead of null for missing text fields
            - Use empty arrays [] instead of null for missing list fields
            - Use 0 instead of null for missing numeric fields
            - Extract ALL available information accurately
            
            Focus on:
            - Clear identification of required vs preferred skills
            - Accurate experience requirements (extract numbers when mentioned)
            - Key responsibilities breakdown
            - Education requirements (if mentioned)
            - Company and role identification
            
            Guidelines:
            - If company name not mentioned, use empty string ""
            - If education level not specified, use "Not specified"
            - If experience not mentioned, use 0
            - Extract skills from context even if not explicitly listed
            - Distinguish between must-have and nice-to-have requirements"""),
            
            ("human", """Analyze this job description and extract requirements:

{job_text}

{format_instructions}

Return ONLY valid JSON format with no additional text. Do NOT use null values.""")
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

    def _create_customization_prompts(self):
        """Create prompts for resume customization and cover letter generation"""
        self.resume_customization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert resume customization specialist. Your task is to customize a resume to better match a specific job description while maintaining truthfulness and authenticity.

Guidelines:
- NEVER fabricate experience, skills, or qualifications
- Focus on reordering, emphasizing, and rewording existing content
- Highlight relevant experience and skills that match the job requirements
- Suggest improvements to summaries and bullet points
- Recommend relevant keywords from the job description
- Maintain professional formatting and structure
- Provide specific, actionable suggestions

Return a JSON response with the following structure:
{
  "customized_summary": "Enhanced professional summary",
  "emphasized_skills": ["skill1", "skill2"],
  "experience_modifications": [
    {
      "section": "work_experience",
      "suggestions": ["suggestion1", "suggestion2"]
    }
  ],
  "keyword_suggestions": ["keyword1", "keyword2"],
  "customization_summary": "Brief explanation of key changes made"
}"""),
            
            ("human", """Customize this resume for the target job:

ORIGINAL RESUME:
{original_resume}

TARGET JOB:
{job_description}

Job Title: {job_title}
Company: {company}
Required Skills: {required_skills}
Experience Required: {experience_required} years

Provide customization suggestions that emphasize relevant experience and skills.""")
        ])

        self.cover_letter_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert cover letter writer. Create compelling, personalized cover letters that highlight the candidate's relevant experience and enthusiasm for the specific role.

Guidelines:
- Write in a professional but engaging tone
- Keep it concise (3-4 paragraphs)
- Start with a strong opening that mentions the specific role
- Highlight 2-3 most relevant qualifications or experiences
- Show knowledge of the company/role
- End with a call to action
- Avoid generic phrases and clichés
- Make it feel personal and authentic

Structure:
1. Opening: Role interest and brief introduction
2. Body: Relevant experience and skills (1-2 paragraphs)
3. Closing: Enthusiasm and next steps

Return the cover letter as plain text."""),
            
            ("human", """Write a personalized cover letter for this candidate and job:

CANDIDATE INFO:
Name: {candidate_name}
Experience: {candidate_experience_years} years
Summary: {candidate_experience}
Key Skills: {candidate_skills}

JOB DETAILS:
Position: {job_title}
Company: {company}
Location: {location}
Required Skills: {required_skills}

Job Requirements:
{job_requirements}

Create a compelling cover letter that connects the candidate's background to this specific opportunity.""")
        ])

        self.customization_analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a career counselor specializing in resume optimization. Analyze a resume against a job description and provide detailed customization recommendations.

Your analysis should identify:
- Skill gaps between the resume and job requirements
- Experience alignment issues
- Missing keywords that should be incorporated
- Priority areas for improvement
- Specific actionable recommendations

Return a JSON response with:
{
  "skill_gaps": ["missing skill 1", "missing skill 2"],
  "experience_recommendations": ["suggestion 1", "suggestion 2"],
  "keyword_suggestions": ["keyword 1", "keyword 2"],
  "priority_changes": [
    {
      "priority": "high/medium/low",
      "change": "specific recommendation",
      "reason": "explanation"
    }
  ],
  "overall_assessment": "summary of candidacy strength"
}"""),
            
            ("human", """Analyze this resume against the job requirements:

RESUME OVERVIEW:
Skills: {resume_skills}
Experience: {resume_experience} years
Summary: {resume_summary}

JOB REQUIREMENTS:
Required Skills: {job_requirements}
Experience Required: {job_experience_required} years
Job Description: {job_description}

Provide detailed customization recommendations to improve the resume's match for this role.""")
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

    def get_resume_customization_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for resume customization"""
        return self.resume_customization_prompt

    def get_cover_letter_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for cover letter generation"""
        return self.cover_letter_prompt

    def get_customization_analysis_prompt(self) -> ChatPromptTemplate:
        """Get prompt template for customization analysis"""
        return self.customization_analysis_prompt

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
