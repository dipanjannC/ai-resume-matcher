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

def test_prompt_manager_customization_prompts_formatting():
    """Test that LangChainPromptManager templates compile without KeyError/ValueError."""
    from app.services.prompt_manager import LangChainPromptManager
    
    pm = LangChainPromptManager()
    
    # Test resume_customization_prompt
    rcp_prompt = pm.get_resume_customization_prompt()
    try:
        formatted_rcp = rcp_prompt.format_messages(
            original_resume="mock resume",
            job_title="mock title",
            company="mock company",
            required_skills="mock skills",
            experience_required="3",
            company_research="mock research",
            job_description="mock description"
        )
        # Should produce 2 messages: system + human
        assert len(formatted_rcp) == 2
        # System message describes JSON output keys in plain text (no literal braces)
        assert "customized_summary" in formatted_rcp[0].content
        # Human message should contain the injected values
        assert "mock resume" in formatted_rcp[1].content
        assert "mock title" in formatted_rcp[1].content
    except Exception as e:
        pytest.fail(f"resume_customization_prompt raised an error during string binding: {e}")

    # Test customization_analysis_prompt
    cap_prompt = pm.get_customization_analysis_prompt()
    try:
        formatted_cap = cap_prompt.format_messages(
            resume_skills="mock resume skills",
            resume_experience="5",
            resume_summary="mock summary",
            job_requirements="mock job requirements",
            job_experience_required="3",
            job_description="mock job description"
        )
        # Should produce 2 messages: system + human
        assert len(formatted_cap) == 2
        # System message describes JSON output keys in plain text (no literal braces)
        assert "skill_gaps" in formatted_cap[0].content
        # Human message should contain the injected values
        assert "mock resume skills" in formatted_cap[1].content
        assert "mock job description" in formatted_cap[1].content
    except Exception as e:
        pytest.fail(f"customization_analysis_prompt raised an error during string binding: {e}")
