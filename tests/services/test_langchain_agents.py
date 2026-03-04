import pytest
import json
from unittest.mock import AsyncMock, patch
from typing import Dict, Any

from app.models.resume_data import ResumeData, JobDescription
from app.models.langchain_models import ResumeParsingOutput, JobParsingOutput

# Import after mocking if necessary, or rely on internal methods
from app.services.langchain_agents import LangChainAgents
from app.core.exceptions import ResumeMatcherException

@pytest.fixture
def mock_langchain_agents():
    # We initialize without API keys since we mock the internal execute method
    with patch('app.services.langchain_agents.LLMService') as mock_llm:
        agents = LangChainAgents()
        return agents


def test_clean_json_response_perfect(mock_langchain_agents):
    """Test cleaning a perfectly formatted JSON string."""
    perfect_json = '{"key": "value", "list": [1, 2, 3]}'
    cleaned = mock_langchain_agents._clean_json_response(perfect_json)
    assert json.loads(cleaned) == {"key": "value", "list": [1, 2, 3]}


def test_clean_json_response_with_markdown(mock_langchain_agents):
    """Test extracting JSON from a markdown block."""
    markdown_json = '''
    Here is the parsed output:
    ```json
    {
        "profile": {"name": "John Doe"},
        "skills": {"technical": ["Python"]}
    }
    ```
    Hope this helps!
    '''
    cleaned = mock_langchain_agents._clean_json_response(markdown_json)
    parsed = json.loads(cleaned)
    assert parsed["profile"]["name"] == "John Doe"


def test_clean_json_null_replacement(mock_langchain_agents):
    """Test that null values are correctly replaced with safe defaults."""
    null_json = '''
    {
        "title": null,
        "experience_years": null,
        "required_skills": null,
        "profile": {
            "name": null
        }
    }
    '''
    cleaned = mock_langchain_agents._clean_json_response(null_json)
    parsed = json.loads(cleaned)
    
    assert parsed["title"] == ""
    assert parsed["experience_years"] == 0
    assert parsed["required_skills"] == []
    assert parsed["profile"]["name"] == ""


def test_clean_json_invalid_structure(mock_langchain_agents):
    """Test handling of completely invalid JSON where fallback returns raw string."""
    invalid = "This is not JSON at all."
    cleaned = mock_langchain_agents._clean_json_response(invalid)
    assert cleaned == invalid


@pytest.mark.asyncio
async def test_parse_resume_success(mock_langchain_agents):
    """Test successful resume parsing with mocked LLM output."""
    mock_response = AsyncMock()
    mock_response.content = '''
    {
        "profile": {
            "name": "Jane Doe",
            "title": "Data Scientist",
            "email": "jane@example.com",
            "phone": "123-456-7890",
            "linkedin": "",
            "location": "NY"
        },
        "experience": {
            "total_years": 4,
            "roles": [],
            "companies": [],
            "responsibilities": [],
            "achievements": []
        },
        "skills": {
            "technical": ["Python", "SQL"],
            "soft": [],
            "certifications": [],
            "languages": []
        },
        "topics": {
            "domains": [],
            "specializations": [],
            "interests": []
        },
        "tools_libraries": {
            "programming_languages": ["Python"],
            "frameworks": [],
            "tools": [],
            "databases": [],
            "cloud_platforms": []
        },
        "summary": "Experienced data scientist.",
        "key_strengths": ["Machine Learning"]
    }
    '''
    
    with patch.object(mock_langchain_agents, '_execute_with_fallback', return_value=mock_response):
        resume_data = await mock_langchain_agents.parse_resume("Raw resume text")
        
        assert isinstance(resume_data, ResumeData)
        assert resume_data.profile.name == "Jane Doe"
        assert resume_data.experience.total_years == 4
        assert "Python" in resume_data.skills.technical


@pytest.mark.asyncio
async def test_parse_job_description_with_nulls(mock_langchain_agents):
    """Test parsing a job description that contains nulls/missing fields from hallucination."""
    mock_response = AsyncMock()
    
    # Simulate a messy output where LLM forgot arrays or put nulls
    mock_response.content = '''
    {
        "title": "Software Engineer",
        "company": null,
        "required_skills": null,
        "experience_years": null,
        "education_level": null,
        "responsibilities": null
    }
    '''
    
    with patch.object(mock_langchain_agents, '_execute_with_fallback', return_value=mock_response):
        job_data = await mock_langchain_agents.parse_job_description("Raw Job Text")
        
        assert isinstance(job_data, JobDescription)
        assert job_data.title == "Software Engineer"
        assert job_data.company == ""  # Replaced null
        assert job_data.required_skills == [] # Replaced null list
        assert job_data.experience_years == 0 # Replaced null int
