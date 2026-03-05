import pytest
from unittest.mock import patch, MagicMock
from app.services.job_scraper import JobScraper
from app.models.resume_data import JobDescription
from app.core.exceptions import ResumeMatcherException

@pytest.fixture
def mock_langchain_agents():
    with patch("app.services.job_scraper.langchain_agents") as mock_agents:
        # Create a mock JobDescription return value
        mock_jd = JobDescription(
            title="Senior Machine Learning Engineer",
            company="ZS",
            required_skills=["Python", "Machine Learning", "LLMs"],
            preferred_skills=["C++"],
            experience_years=5,
            education_level="Master's",
            responsibilities=["Build AI models", "Deploy to production"]
        )
        async def mock_parse(*args, **kwargs):
            return mock_jd
            
        mock_agents.parse_job_description.side_effect = mock_parse
        yield mock_agents

@pytest.fixture
def job_scraper():
    return JobScraper()

def test_extract_raw_text(job_scraper):
    """Test that BeautifulSoup properly rips out scripts and noisy tags."""
    html_input = '''
        <html>
            <head>
                <script>alert("noisy script")</script>
                <style>.hidden { display: none; }</style>
            </head>
            <body>
                <nav>Navigation Header Skip to Main Content</nav>
                <h1>Senior ML Engineer</h1>
                <p>We are looking for a python developer with 5 years experience.</p>
                <!-- Some noise -->
                <script>track_analytics()</script>
            </body>
        </html>
    '''
    
    clean_text = job_scraper._extract_raw_text(html_input)
    
    assert "noisy script" not in clean_text
    assert ".hidden" not in clean_text
    assert "track_analytics" not in clean_text
    assert "Senior ML Engineer" in clean_text
    assert "python developer with 5 years experience" in clean_text

@pytest.mark.asyncio
async def test_scrape_and_parse_success(job_scraper, mock_langchain_agents):
    """Test the full flow where extraction passes valid text to LangChain."""
    
    # Mock the HTTP request with enough characters to pass the > 50 chars validation
    mock_html = "<html><body><h1>Test Engineer</h1><p>We are currently looking for a Senior Developer who has at least 5 years of Python experience and is proficient in Machine Learning.</p></body></html>"
    with patch.object(job_scraper, '_fetch_page', return_value=mock_html):
        result = await job_scraper.scrape_and_parse("http://example.com/job/123")
        
    assert result.title == "Senior Machine Learning Engineer"
    assert result.company == "ZS"
    assert "Python" in result.required_skills
    # Ensure langchain was called with the scraped text
    mock_langchain_agents.parse_job_description.assert_called_once()
    
def test_fetch_page_invalid_url(job_scraper):
    """Test URL validation failures."""
    with pytest.raises(ResumeMatcherException, match="Failed to fetch job posting"):
        job_scraper._fetch_page("not-a-url")

@pytest.mark.asyncio
async def test_scrape_and_parse_too_short(job_scraper):
    """Test that extremely short content fails early to save LLM tokens."""
    with patch.object(job_scraper, '_fetch_page', return_value="<html><body><p>No</p></body></html>"):
        with pytest.raises(ResumeMatcherException, match="too short"):
            await job_scraper.scrape_and_parse("http://example.com")
