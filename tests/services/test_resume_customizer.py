import pytest
from unittest.mock import AsyncMock, patch
from app.models.resume_data import ResumeData, JobDescription
from app.services.resume_customizer import ResumeCustomizerService

@pytest.fixture
def mock_resume_data():
    return ResumeData(
        id="res-123",
        summary="An experienced software engineer."
    )

@pytest.fixture
def mock_job_data():
    job = JobDescription()
    job.title = "Senior Developer"
    job.company = "TechCorp"
    return job

@pytest.fixture
def customizer():
    # Patch dependencies so we can instantiate the service safely
    with patch('app.services.resume_customizer.langchain_agents') as m_agents, \
         patch('app.services.resume_customizer.prompt_manager') as m_prompts:
        service = ResumeCustomizerService()
        return service

@pytest.mark.asyncio
async def test_customize_resume_formatting(customizer, mock_resume_data, mock_job_data):
    """Test generating a customized resume returns cleanly structured dict."""
    
    # Mock the returned dictionary from LangchainAgents (after clean JSON parsing)
    mock_customized_dict = {
        "customized_resume": {
            "summary": "Tailored summary for TechCorp",
            "experience": []
        },
        "agentic_reasoning": "I added keywords because TechCorp values XYZ."
    }
    
    customizer.langchain_agents.customize_resume_for_job = AsyncMock(return_value=mock_customized_dict)
    
    result = await customizer.customize_resume(mock_resume_data, mock_job_data)
    
    assert result["success"] is True
    assert result["job_title"] == "Senior Developer"
    assert result["company"] == "TechCorp"
    assert "agentic_reasoning" in result
    
@pytest.mark.asyncio
async def test_customize_resume_failure(customizer, mock_resume_data, mock_job_data):
    """Test handling of customization failure."""
    customizer.langchain_agents.customize_resume_for_job = AsyncMock(return_value=None)
    
    result = await customizer.customize_resume(mock_resume_data, mock_job_data)
    
    assert result["success"] is False
    assert "error" in result
