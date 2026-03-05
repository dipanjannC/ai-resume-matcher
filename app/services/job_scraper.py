"""
Robust Job Description Scraper

Extracts raw text from job URLs and uses the LLM parser to cleanly structure
messy HTML dumps, ignoring navigation noise and boilerplate.
"""

import requests
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
import urllib3
from urllib3.exceptions import InsecureRequestWarning

from app.services.langchain_agents import langchain_agents
from app.models.resume_data import JobDescription
from app.core.logging import get_logger
from app.core.exceptions import ResumeMatcherException

# Suppress SSL warnings
urllib3.disable_warnings(InsecureRequestWarning)

logger = get_logger(__name__)


class JobScraper:
    """Service for robustly scraping and parsing Job Descriptions using LLMs"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }

    def _fetch_page(self, url: str) -> str:
        """Fetch raw HTML from URL safely"""
        try:
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError("Invalid URL format. Check http:// or https://")

            try:
                # First try with SSL verification
                response = requests.get(url, headers=self.headers, timeout=20, allow_redirects=True)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                # Fallback without SSL verification
                response = requests.get(url, headers=self.headers, timeout=20, allow_redirects=True, verify=False)
                response.raise_for_status()

            if len(response.text) < 200:
                raise ValueError("URL returned insufficient content. Page may be protected.")

            return response.text

        except Exception as e:
            logger.error(f"Failed to fetch {url}: {e}")
            raise ResumeMatcherException(f"Failed to fetch job posting: {str(e)}")

    def _extract_raw_text(self, html_content: str) -> str:
        """Extract visible text from HTML dump, ignoring scripts and styles"""
        soup = BeautifulSoup(html_content, "html.parser")

        # Destroy script and style tags
        for script in soup(["script", "style", "noscript"]):
            script.decompose()

        # Extract text collapsing extra whitespace
        text = soup.get_text(separator=' ', strip=True)
        
        # Additional cleanup to prevent token overflow and reduce LLM noise
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Typical ATS pages can have thousands of characters of boilerplate.
        # Limit to the most crucial 15000 chars to avoid breaking the LLM context window
        if len(text) > 15000:
            text = text[:15000] + "..."

        return text

    async def scrape_and_parse(self, url: str) -> JobDescription:
        """
        Fetch a URL, extract messy text, and pass it directly to the LangChain 
        agents to intelligently extract requirements and ignore boilerplate.
        """
        logger.info(f"Scraping job URL: {url}")
        
        # 1. Fetch HTML
        html_content = self._fetch_page(url)
        
        # 2. Rip raw text (will include messy tags, navigation menus, etc.)
        raw_text = self._extract_raw_text(html_content)
        
        if len(raw_text) < 50:
            raise ResumeMatcherException("Extracted text is too short to be a valid job description.")
            
        logger.info(f"Extracted {len(raw_text)} characters of raw text. Relaying to LangChain...")
        
        # 3. Leverage the highly-robust LangChainAgents to structure the data
        # The LLM is smart enough to ignore "Skip to Main Content", "Terms of Use", etc.
        job_data = await langchain_agents.parse_job_description(raw_text)
        
        # Ensure the URL is tracked
        job_data.raw_text = f"Source URL: {url}\n\n{raw_text}"
        
        return job_data


# Global instance
job_scraper = JobScraper()
