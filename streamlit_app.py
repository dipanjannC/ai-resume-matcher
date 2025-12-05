#!/usr/bin/env python3
"""
AI Resume Matcher - Streamlit Web Interface
Interactive web application for resume upload, job matching, and visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any
import requests
import re
import json
from urllib.parse import urlparse
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Suppress SSL warnings when using verify=False as fallback
urllib3.disable_warnings(InsecureRequestWarning)

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.resume_processor import resume_processor
from app.services.job_processor import job_processor
from app.services.data_pipeline import data_pipeline
from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service
from app.services.resume_customizer import resume_customizer
from app.models.resume_data import ResumeData, JobDescription, MatchResult
from app.core.logging import get_logger
from app.utils.pdf_generator import generate_resume_pdf
from io import BytesIO
import zipfile

logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Resume Matcher",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stored_jobs():
    """Get stored jobs with caching"""
    try:
        return asyncio.run(job_processor.list_stored_jobs())
    except Exception as e:
        logger.error(f"Error loading jobs: {e}")
        return []

@st.cache_data(ttl=300)  # Cache for 5 minutes  
def get_processed_resumes():
    """Get processed resumes with caching"""
    try:
        return asyncio.run(resume_processor.list_processed_resumes())
    except Exception as e:
        logger.error(f"Error loading resumes: {e}")
        return []

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_pipeline_stats():
    """Get pipeline statistics with caching"""
    try:
        return data_pipeline.get_pipeline_stats()
    except Exception as e:
        logger.error(f"Error loading stats: {e}")
        return {"total_resumes": 0, "total_jobs": 0}

def initialize_data():
    """Initialize sample data if database is empty"""
    if not st.session_state.initialized:
        with st.spinner("Initializing system..."):
            stats = get_pipeline_stats()
            
            # Load sample data if database is empty
            if stats['total_jobs'] == 0:
                st.info("Loading sample data for demonstration...")
                try:
                    result = asyncio.run(data_pipeline.process_sample_data())
                    st.success(f"✅ Loaded {result['jobs']['processed']} sample jobs!")
                    
                    # Clear cache to reflect new data
                    st.cache_data.clear()
                    
                except Exception as e:
                    st.error(f"Failed to load sample data: {e}")
            
            st.session_state.initialized = True

class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.resume_processor = resume_processor
        self.job_processor = job_processor
        self.data_pipeline = data_pipeline
    
    def extract_job_from_url(self, url):
        """Enhanced job description extraction from URL"""
        try:
            # Validate URL
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return {"error": "Invalid URL format. Please include http:// or https://"}
            
            # Enhanced headers to mimic different browsers and avoid blocking
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }
            
            # Make request with extended timeout and better error handling
            try:
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
                response.raise_for_status()
            except requests.exceptions.SSLError:
                # Retry without SSL verification as fallback
                response = requests.get(url, headers=headers, timeout=15, allow_redirects=True, verify=False)
                response.raise_for_status()
            
            # Check if we got actual content
            if len(response.text) < 100:
                return {"error": "URL returned very little content. The page might be protected or require login."}
            
            # Extract text content with enhanced processing
            text_content = self.extract_text_from_html(response.text)
            
            # Clean and filter relevant content
            cleaned_content = self.clean_job_content(text_content)
            
            # Extract job details using enhanced patterns
            job_info = self.parse_job_details_enhanced(response.text, cleaned_content, url)
            
            # Ensure we have meaningful content (be more lenient)
            if len(cleaned_content.strip()) < 20:
                # If we have job title/company but little content, use original text
                if job_info.get("title") or job_info.get("company"):
                    cleaned_content = text_content[:2000]  # Use first 2000 chars of original
                    if not cleaned_content.strip():
                        cleaned_content = f"Job Title: {job_info.get('title', 'N/A')}\nCompany: {job_info.get('company', 'N/A')}\n\nJob details extracted from: {url}"
                else:
                    return {"error": "Could not extract meaningful job content from this URL. The page might be dynamically loaded, require login, or use JavaScript to display content."}
            
            return {
                "success": True,
                "title": job_info.get("title", ""),
                "company": job_info.get("company", ""),
                "description": cleaned_content,
                "location": job_info.get("location", ""),
                "experience": job_info.get("experience", ""),
                "url": url
            }
            
        except requests.exceptions.Timeout:
            return {"error": "Request timed out (15s). The website might be slow or unavailable."}
        except requests.exceptions.ConnectionError:
            return {"error": "Could not connect to the website. Please check the URL and your internet connection."}
        except requests.exceptions.HTTPError as e:
            return {"error": f"HTTP Error {e.response.status_code}: The webpage is not accessible. It might require login or be restricted."}
        except requests.exceptions.RequestException as e:
            return {"error": f"Network error: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error processing URL: {str(e)}"}
    
    def extract_text_from_html(self, html_content):
        """Enhanced text extraction from HTML content"""
        # Remove unwanted tags and their content
        unwanted_tags = ['script', 'style', 'nav', 'header', 'footer', 'aside', 'noscript']
        for tag in unwanted_tags:
            html_content = re.sub(f'<{tag}[^>]*>.*?</{tag}>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Convert some HTML elements to meaningful text
        html_content = re.sub(r'<br[^>]*>', '\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<p[^>]*>', '\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</p>', '\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<div[^>]*>', '\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</div>', '\n', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'<li[^>]*>', '\n• ', html_content, flags=re.IGNORECASE)
        html_content = re.sub(r'</li>', '\n', html_content, flags=re.IGNORECASE)
        
        # Remove remaining HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Decode HTML entities (extended list)
        html_entities = {
            '&nbsp;': ' ', '&amp;': '&', '&lt;': '<', '&gt;': '>', 
            '&quot;': '"', '&#39;': "'", '&apos;': "'", '&cent;': '¢', 
            '&pound;': '£', '&yen;': '¥', '&euro;': '€', '&copy;': '©', 
            '&reg;': '®', '&trade;': '™', '&#8211;': '–', '&#8212;': '—',
            '&#8216;': ''', '&#8217;': ''', '&#8220;': '"', '&#8221;': '"',
            '&#8226;': '•', '&#8230;': '…', '&mdash;': '—', '&ndash;': '–',
            '&rsquo;': "'", '&lsquo;': "'", '&rdquo;': '"', '&ldquo;': '"',
            '&hellip;': '…', '&bull;': '•'
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Clean up whitespace and formatting
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'\n\s*\n', '\n', text)  # Multiple newlines to single
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Remove excessive whitespace while preserving some structure
        lines = text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 2:  # Keep lines with meaningful content
                clean_lines.append(line)
        
        return '\n'.join(clean_lines)
    
    def clean_job_content(self, text_content):
        """Clean and filter job-relevant content"""
        # Remove common website navigation and footer content
        lines = text_content.split('\n')
        filtered_lines = []
        
        skip_patterns = [
            r'cookie', r'privacy policy', r'terms', r'subscribe', r'newsletter',
            r'follow us', r'social media', r'linkedin', r'twitter', r'facebook',
            r'navigation', r'menu', r'footer', r'header', r'sidebar',
            r'advertisement', r'ad ', r'sponsored', r'related jobs'
        ]
        
        for line in lines:
            line = line.strip()
            if len(line) < 10:  # Skip very short lines
                continue
            
            # Skip lines that match common non-job patterns
            if any(re.search(pattern, line.lower()) for pattern in skip_patterns):
                continue
            
            # Skip lines that are mostly punctuation or numbers
            if len(re.sub(r'[^a-zA-Z]', '', line)) < len(line) * 0.5:
                continue
                
            filtered_lines.append(line)
        
        # Join and limit content length
        cleaned_content = '\n'.join(filtered_lines)
        
        # Limit to reasonable length (about 5000 characters for job descriptions)
        if len(cleaned_content) > 5000:
            cleaned_content = cleaned_content[:5000] + "...\n\n[Content truncated for processing]"
        
        return cleaned_content
    
    def parse_job_details_enhanced(self, html_content, text_content, url):
        """Enhanced job detail parsing with multiple strategies"""
        job_info = {"title": "", "company": "", "location": "", "experience": ""}
        
        # Strategy 1: Extract from HTML title tag
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
        if title_match:
            title_text = title_match.group(1).strip()
            # Common patterns in job posting titles
            patterns = [
                r'(.+?)\s*[-|–]\s*(.+?)\s*[-|–]\s*(.+)',  # Title - Company - Site
                r'(.+?)\s*[-|–]\s*(.+)',  # Title - Company
                r'(.+?)\s*at\s*(.+)',  # Title at Company
            ]
            
            for pattern in patterns:
                match = re.search(pattern, title_text, re.IGNORECASE)
                if match:
                    job_info["title"] = match.group(1).strip()
                    job_info["company"] = match.group(2).strip()
                    break
            
            # If no pattern matched, use whole title
            if not job_info["title"]:
                job_info["title"] = title_text
        
        # Strategy 2: Enhanced text content parsing
        enhanced_patterns = {
            "title": [
                r'job title[:\s]*([^\n\r]+)',
                r'position[:\s]*([^\n\r]+)',
                r'role[:\s]*([^\n\r]+)',
                r'we are hiring[:\s]*([^\n\r]+)',
                r'looking for[:\s]*([^\n\r]+)',
                r'join our team as[:\s]*([^\n\r]+)',
                r'<h1[^>]*>([^<]+)</h1>',  # H1 tags often contain job titles
                r'<h2[^>]*>([^<]+)</h2>',  # H2 tags as fallback
            ],
            "company": [
                r'company[:\s]*([^\n\r]+)',
                r'employer[:\s]*([^\n\r]+)',
                r'organization[:\s]*([^\n\r]+)',
                r'about\s+([a-zA-Z][a-zA-Z\s&\.]+?)(?:\n|is|was)',
                r'join\s+([a-zA-Z][a-zA-Z\s&\.]+?)(?:\n|team|today)',
                r'at\s+([a-zA-Z][a-zA-Z\s&\.]+?)(?:\n|,|\.|we)',
            ],
            "location": [
                r'location[:\s]*([^\n\r]+)',
                r'based in[:\s]*([^\n\r]+)',
                r'office[:\s]*([^\n\r]+)',
                r'remote',
                r'work from home',
                r'([a-zA-Z\s]+,\s*[A-Z]{2,})',  # City, State/Country pattern
            ],
            "experience": [
                r'(\d+)[\+\-\s]*years?\s*(?:of\s*)?experience',
                r'experience[:\s]*(\d+[\+\-\s]*years?)',
                r'minimum\s*(\d+\s*years?)',
                r'(\d+)\s*to\s*(\d+)\s*years?',
            ]
        }
        
        # Apply enhanced patterns
        for field, patterns in enhanced_patterns.items():
            if job_info.get(field):  # Skip if already found
                continue
                
            for pattern in patterns:
                # Try HTML first, then text content
                for content in [html_content, text_content]:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        if field == "experience" and len(match.groups()) > 1:
                            # Handle range patterns
                            job_info[field] = f"{match.group(1)}-{match.group(2)} years"
                        else:
                            job_info[field] = match.group(1).strip()
                        break
                if job_info.get(field):
                    break
        
        # Strategy 3: JSON-LD structured data extraction
        json_ld_match = re.search(r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', 
                                 html_content, re.IGNORECASE | re.DOTALL)
        if json_ld_match:
            try:
                json_data = json.loads(json_ld_match.group(1))
                if isinstance(json_data, dict):
                    # Common schema.org JobPosting structure
                    if json_data.get('@type') == 'JobPosting' or 'JobPosting' in str(json_data.get('@type', '')):
                        if not job_info["title"] and json_data.get('title'):
                            job_info["title"] = json_data['title']
                        if not job_info["company"] and json_data.get('hiringOrganization', {}).get('name'):
                            job_info["company"] = json_data['hiringOrganization']['name']
                        if not job_info["location"] and json_data.get('jobLocation', {}).get('address'):
                            location_data = json_data['jobLocation']['address']
                            if isinstance(location_data, dict):
                                location_parts = []
                                if location_data.get('addressLocality'):
                                    location_parts.append(location_data['addressLocality'])
                                if location_data.get('addressRegion'):
                                    location_parts.append(location_data['addressRegion'])
                                job_info["location"] = ", ".join(location_parts)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass  # Ignore JSON parsing errors
        
        # Clean up extracted data
        for field in job_info:
            if job_info[field]:
                # Remove extra whitespace and clean up
                job_info[field] = re.sub(r'\s+', ' ', job_info[field]).strip()
                # Remove common prefixes/suffixes
                job_info[field] = re.sub(r'^(job\s*)?title[:\s]*', '', job_info[field], flags=re.IGNORECASE)
                job_info[field] = re.sub(r'^company[:\s]*', '', job_info[field], flags=re.IGNORECASE)
                # Limit length
                if len(job_info[field]) > 100:
                    job_info[field] = job_info[field][:100] + "..."
        
        # Fallback: extract company from domain
        if not job_info["company"]:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            company_from_domain = domain.split('.')[0].replace('-', ' ').title()
            job_info["company"] = company_from_domain
        
        return job_info
    
    def parse_job_details(self, text_content, url):
        """Parse job title and company from text content"""
        job_info = {"title": "", "company": ""}
        
        # Common job title patterns
        title_patterns = [
            r'<title[^>]*>([^<]*?)\s*[-|–]\s*([^<]*)</title>',  # HTML title
            r'Job Title[:\s]*([^\n\r]+)',
            r'Position[:\s]*([^\n\r]+)',
            r'Role[:\s]*([^\n\r]+)',
            r'We are looking for[:\s]*([^\n\r]+)',
            r'Join us as[:\s]*([^\n\r]+)'
        ]
        
        # Company patterns
        company_patterns = [
            r'Company[:\s]*([^\n\r]+)',
            r'Organization[:\s]*([^\n\r]+)',
            r'Employer[:\s]*([^\n\r]+)',
            r'About ([^\n\r]+?)(?:\n|We are|Founded)',
            r'At ([^\n\r,]+)'
        ]
        
        # Try to extract title
        for pattern in title_patterns:
            match = re.search(pattern, text_content, re.IGNORECASE)
            if match:
                if len(match.groups()) > 1:
                    # Title with company in HTML title
                    job_info["title"] = match.group(1).strip()
                    job_info["company"] = match.group(2).strip()
                else:
                    job_info["title"] = match.group(1).strip()
                break
        
        # Try to extract company if not found
        if not job_info["company"]:
            for pattern in company_patterns:
                match = re.search(pattern, text_content, re.IGNORECASE)
                if match:
                    job_info["company"] = match.group(1).strip()
                    break
        
        # Clean up extracted data
        if job_info["title"]:
            job_info["title"] = re.sub(r'\s+', ' ', job_info["title"]).strip()
        if job_info["company"]:
            job_info["company"] = re.sub(r'\s+', ' ', job_info["company"]).strip()
        
        # Fallback: extract from URL domain
        if not job_info["company"]:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            job_info["company"] = domain.split('.')[0].title()
        
        return job_info
        
    def render_sidebar(self):
        """Render the sidebar with navigation and core actions"""
        with st.sidebar:
            st.title("🤖 AI Resume Matcher")
            st.markdown("---")
            
            # --- API Configuration ---
            with st.expander("🔑 API Configuration", expanded=False):
                import os
                
                # OpenAI
                openai_key = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""), key="sidebar_openai_key")
                if openai_key:
                    os.environ["OPENAI_API_KEY"] = openai_key
                
                # Gemini
                gemini_key = st.text_input("Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", ""), key="sidebar_gemini_key")
                if gemini_key:
                    os.environ["GEMINI_API_KEY"] = gemini_key
                    
                # Groq
                groq_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""), key="sidebar_groq_key")
                if groq_key:
                    os.environ["GROQ_API_KEY"] = groq_key
            
            # --- Navigation ---
            st.header("Navigation")
            
            # Handle page selection with session state to allow programmatic navigation
            if 'selected_main_page' not in st.session_state:
                st.session_state.selected_main_page = "📄 Resume Upload"
                
            page = st.radio(
                "Choose a page:",
                ["📄 Resume Upload", "📋 Job Management", "🎯 Job Matching", "✏️ Resume Customizer", "📊 Analytics"],
                key="nav_radio",
                index=["📄 Resume Upload", "📋 Job Management", "🎯 Job Matching", "✏️ Resume Customizer", "📊 Analytics"].index(st.session_state.selected_main_page) if st.session_state.selected_main_page in ["📄 Resume Upload", "📋 Job Management", "🎯 Job Matching", "✏️ Resume Customizer", "📊 Analytics"] else 0
            )
            
            # Update session state if changed via radio
            if page != st.session_state.selected_main_page:
                st.session_state.selected_main_page = page
                st.rerun()
            
            st.markdown("---")
            
            # --- Contextual Sidebar Content ---
            if page == "📄 Resume Upload":
                self.sidebar_resume_upload()
            elif page == "📋 Job Management":
                self.sidebar_add_job()
                
            return page

    def sidebar_resume_upload(self):
        """Sidebar component for resume upload"""
        st.header("📤 Upload Resumes")
        
        uploaded_files = st.file_uploader(
            "Choose resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Supported: PDF, DOCX, TXT",
            key="sidebar_uploader"
        )
        
        if uploaded_files:
            st.session_state.sidebar_uploaded_files = uploaded_files
            
            st.markdown("### ⚙️ Options")
            st.checkbox("Batch process", value=True, key="sidebar_batch_process")
            st.checkbox("Overwrite existing", value=False, key="sidebar_overwrite")
            
            if st.button("🚀 Process Resumes", type="primary", use_container_width=True):
                st.session_state.trigger_processing = True
        else:
            if 'sidebar_uploaded_files' in st.session_state:
                del st.session_state.sidebar_uploaded_files

    def sidebar_add_job(self):
        """Sidebar component for adding jobs"""
        st.header("➕ Add New Job")
        
        # We'll implement the full form logic here or call a helper
        # For now, let's just put a placeholder or the actual form if it fits
        # Since the form is complex, we might want to keep it simple here or move the whole form
        
        with st.expander("🔗 Import from URL", expanded=True):
            job_url = st.text_input("Job URL", placeholder="https://...", key="sidebar_job_url")
            if st.button("🤖 Extract", key="sidebar_extract_btn", use_container_width=True):
                if job_url:
                    with st.spinner("Extracting..."):
                        extraction_result = self.extract_job_from_url(job_url)
                        if extraction_result.get("success"):
                            st.session_state.url_job_title = extraction_result.get("title", "")
                            st.session_state.url_company = extraction_result.get("company", "")
                            st.session_state.url_description = extraction_result.get("description", "")
                            st.session_state.url_location = extraction_result.get("location", "")
                            st.session_state.url_experience = extraction_result.get("experience", "")
                            st.success("Extracted!")
                        else:
                            st.error("Failed to extract")
        
        st.markdown("---")
        st.markdown("**📝 Job Details**")
        
        with st.form("sidebar_job_form", clear_on_submit=True):
            job_title = st.text_input("Job Title", value=getattr(st.session_state, 'url_job_title', ''))
            company = st.text_input("Company", value=getattr(st.session_state, 'url_company', ''))
            location = st.text_input("Location", value=getattr(st.session_state, 'url_location', ''))
            experience = st.number_input("Experience (yrs)", min_value=0, value=3)
            
            description = st.text_area("Description", value=getattr(st.session_state, 'url_description', ''), height=150)
            
            if st.form_submit_button("💾 Save Job", type="primary", use_container_width=True):
                if job_title and description:
                    self.save_job_description(job_title, company, description, experience, location, "")
                    st.success("Job saved!")
                    # Clear session state
                    for key in ['url_job_title', 'url_company', 'url_description', 'url_location', 'url_experience']:
                        if hasattr(st.session_state, key):
                            delattr(st.session_state, key)
                else:
                    st.error("Title & Desc required")

    def run(self):
        """Main application entry point"""
        # Initialize data if needed
        initialize_data()
        
        # Render sidebar and get selected page
        page = self.render_sidebar()
        
        # Main content area
        if page == "📄 Resume Upload":
            self.resume_upload_page()
        elif page == "📋 Job Management":
            self.job_management_page()
        elif page == "🎯 Job Matching":
            self.job_matching_page()
        elif page == "✏️ Resume Customizer":
            self.resume_customizer_page()
        elif page == "📊 Analytics":
            self.analytics_page()

    
    
    def resume_upload_page(self):
        """Resume upload and processing page with improved validation"""
        st.header("📄 Resume Upload & Processing")
        
        # System status check
        self.check_system_status()
        
        # Check for files from sidebar
        uploaded_files = st.session_state.get('sidebar_uploaded_files', [])
        
        if not uploaded_files:
            st.info("👈 Please upload resume files using the sidebar.")
            
            # Show supported formats info
            st.markdown("""
            ### Supported Formats
            • **PDF** (.pdf)
            • **Word** (.docx)
            • **Text** (.txt)
            
            ### Instructions
            1. Use the **Upload Resumes** section in the sidebar
            2. Select one or multiple files
            3. Click **Process Resumes** to start analysis
            """)
        else:
            st.markdown(f"### 📋 Selected Files ({len(uploaded_files)})")
            
            # File validation and preview
            valid_files = []
            
            for file in uploaded_files:
                file_size = len(file.getbuffer())
                file_size_mb = file_size / (1024 * 1024)
                
                if file_size_mb > 10:  # 10MB limit
                    st.error(f"❌ {file.name}: File too large ({file_size_mb:.1f}MB). Maximum size is 10MB.")
                elif file_size < 100:  # Minimum size check
                    st.error(f"❌ {file.name}: File too small ({file_size} bytes). File might be empty.")
                else:
                    st.success(f"✅ {file.name}: Valid ({file_size_mb:.1f}MB)")
                    valid_files.append(file)
            
            if valid_files:
                # Check if processing triggered
                if st.session_state.get('trigger_processing', False):
                    # Reset trigger
                    st.session_state.trigger_processing = False
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("📊 Processing Status")
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                    
                    # Get options from sidebar keys
                    batch_process = st.session_state.get("sidebar_batch_process", True)
                    # We don't have show_detailed_errors in sidebar, default to True
                    show_detailed_errors = True
                    
                    self.process_uploaded_resumes(valid_files, progress_bar, status_text, batch_process, show_detailed_errors)
            else:
                st.warning("⚠️ No valid files to process. Please check file formats and sizes.")
        
        # Display processed resumes
        st.divider()
        self.display_processed_resumes()

    def check_system_status(self):
        """Check if system is ready for processing"""
        try:
            # Check if required directories exist
            from pathlib import Path
            data_dir = Path("data")
            temp_dir = data_dir / "temp"
            
            # Create directories if they don't exist
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            # Check environment variables
            import os
            if not os.getenv('OPENAI_API_KEY') and not os.getenv('GROQ_API_KEY'):
                st.warning("⚠️ **API Key Missing**: Set OPENAI_API_KEY or GROQ_API_KEY environment variable for AI processing.")
                
            # Show system ready status
            with st.expander("🔧 System Status", expanded=False):
                st.write("**Data Directory:** ✅ Available")
                st.write("**Temp Directory:** ✅ Created")
                api_status = "✅ Available" if (os.getenv('OPENAI_API_KEY') or os.getenv('GROQ_API_KEY')) else "❌ Missing"
                st.write(f"**API Key:** {api_status}")
                
        except Exception as e:
            st.error(f"System check failed: {str(e)}")
    
    def job_management_page(self):
        """Job description management page with clean UI"""
        st.header("📋 Job Management")
        
        # Create tabs for better organization
        tab1, tab2 = st.tabs(["➕ Add New Job", "📋 Manage Jobs"])
        
        with tab1:
            self.add_job_form()
        
        with tab2:
            self.display_jobs_table()
    
    def job_matching_page(self):
        """Enhanced job search and filtering page"""
        st.header("🎯 Smart Job Search & Matching")
        st.markdown("**Find the perfect job opportunities and see how you match up**")
        
        # Get stored jobs
        stored_jobs = get_stored_jobs()
        
        if not stored_jobs:
            st.warning("No job opportunities available. Please add jobs in the Job Management page first.")
            col1, col2, col3 = st.columns(3)
            with col2:
                if st.button("➕ Add Jobs Now", use_container_width=True):
                    st.session_state.selected_main_page = "� Job Management"
                    st.rerun()
            return
        
        # Create tabs for different search approaches  
        tab1, tab2, tab3 = st.tabs([
            "🔍 Search Jobs", 
            "🎯 Match Jobs to My Profile", 
            "📊 Job Market Insights"
        ])
        
        with tab1:
            self.job_search_interface(stored_jobs)
        
        with tab2:
            self.profile_job_matching(stored_jobs)
        
        with tab3:
            self.job_market_insights(stored_jobs)
    
    def job_search_interface(self, stored_jobs):
        """Enhanced job search and filtering interface"""
        st.subheader("🔍 Search & Filter Jobs")
        
        # Search and filter controls
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Main search bar
            search_query = st.text_input(
                "Search Jobs",
                placeholder="Search by job title, company, skills, or keywords...",
                help="Enter keywords to find relevant job opportunities"
            )
        
        with col2:
            st.write("**Quick Filters:**")
            show_all = st.checkbox("Show All Jobs", value=True)
        
        # Advanced filters
        with st.expander("🔧 Advanced Filters", expanded=not show_all):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Experience filter
                min_exp = st.number_input("Min Experience (years)", min_value=0, max_value=20, value=0)
                max_exp = st.number_input("Max Experience (years)", min_value=0, max_value=20, value=20)
            
            with col2:
                # Company filter
                all_companies = set(job.get('company', '').strip() for job in stored_jobs if job.get('company', '').strip())
                selected_companies = st.multiselect("Companies", sorted(list(all_companies)))
                
                # Location filter
                all_locations = set(job.get('location', '').strip() for job in stored_jobs if job.get('location', '').strip())
                selected_locations = st.multiselect("Locations", sorted(list(all_locations)))
            
            with col3:
                # Skills filter
                all_skills = set()
                for job in stored_jobs:
                    skills = job.get('required_skills', []) + job.get('preferred_skills', [])
                    for skill in skills:
                        if skill and skill.strip():
                            all_skills.add(skill.strip())
                
                selected_skills = st.multiselect("Required Skills", sorted(list(all_skills)))
                skills_match_all = st.checkbox("Must have ALL selected skills", value=False)
        
        # Apply filters
        filtered_jobs = self.filter_jobs(
            stored_jobs, search_query, min_exp, max_exp, 
            selected_companies, selected_locations, selected_skills, skills_match_all
        )
        
        # Display results
        if filtered_jobs:
            st.markdown(f"### 📋 Found {len(filtered_jobs)} Job Opportunities")
            
            # Sort options
            col1, col2 = st.columns([3, 1])
            with col2:
                sort_by = st.selectbox("Sort by:", ["Recently Added", "Experience Required", "Company", "Job Title"])
            
            # Sort jobs
            if sort_by == "Recently Added":
                filtered_jobs.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            elif sort_by == "Experience Required":
                filtered_jobs.sort(key=lambda x: x.get('experience_years', 0))
            elif sort_by == "Company":
                filtered_jobs.sort(key=lambda x: x.get('company', ''))
            elif sort_by == "Job Title":
                filtered_jobs.sort(key=lambda x: x.get('title', ''))
            
            # Display job cards
            for i, job in enumerate(filtered_jobs):
                self.display_job_card(job, i)
        else:
            st.warning("No jobs match your search criteria. Try adjusting your filters.")
    
    def profile_job_matching(self, stored_jobs):
        """Match jobs to user profile"""
        st.subheader("🎯 Find Jobs That Match Your Profile")
        st.markdown("*Select your profile to see how you match with available jobs*")
        
        # Get processed resumes
        try:
            processed_resumes = asyncio.run(self.resume_processor.list_processed_resumes())
            
            if not processed_resumes:
                st.warning("No resumes processed yet. Please upload and process your resume first.")
                if st.button("📄 Upload Resume Now"):
                    st.session_state.selected_main_page = "📄 Resume Upload"
                    st.rerun()
                return
            
            # Resume selection
            resume_options = {}
            for resume in processed_resumes:
                display_name = f"{resume.get('filename', 'Unknown')}"
                if resume.get('profile', {}).get('name'):
                    display_name = f"{resume['profile']['name']} - {resume.get('filename', 'Unknown')}"
                resume_options[display_name] = resume['id']
            
            selected_resume = st.selectbox(
                "Select Your Profile:",
                list(resume_options.keys()),
                help="Choose which resume profile to match against jobs"
            )
            
            if selected_resume:
                resume_id = resume_options[selected_resume]
                
                # Matching settings
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    st.markdown("**Matching Settings:**")
                    match_threshold = st.slider("Match Threshold", 0.0, 1.0, 0.3, 0.1, help="Minimum match score to show")
                    max_results = st.slider("Max Results", 5, 50, 15)
                
                with col1:
                    if st.button("🎯 Find Matching Jobs", type="primary", use_container_width=True):
                        self.perform_profile_job_matching(resume_id, stored_jobs, match_threshold, max_results)
                
                # Quick match options (separate row to avoid nesting)
                st.markdown("**Quick Options:**")
                quick_col1, quick_col2 = st.columns(2)
                with quick_col1:
                    if st.button("🏃 Quick Match (Top 5)", use_container_width=True):
                        self.perform_profile_job_matching(resume_id, stored_jobs, 0.2, 5)
                with quick_col2:
                    if st.button("🔍 Detailed Match (All)", use_container_width=True):
                        self.perform_profile_job_matching(resume_id, stored_jobs, 0.0, len(stored_jobs))
        
        except Exception as e:
            st.error(f"Error loading resumes: {str(e)}")
    
    def job_market_insights(self, stored_jobs):
        """Show job market insights"""
        st.subheader("📊 Job Market Intelligence")
        
        # Market overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Opportunities", len(stored_jobs))
        
        with col2:
            companies = set(job.get('company', '') for job in stored_jobs if job.get('company', '').strip())
            st.metric("Hiring Companies", len(companies))
        
        with col3:
            avg_exp = sum(job.get('experience_years', 0) for job in stored_jobs) / len(stored_jobs)
            st.metric("Avg Experience Req.", f"{avg_exp:.1f} years")
        
        with col4:
            remote_jobs = sum(1 for job in stored_jobs if 'remote' in job.get('location', '').lower() or not job.get('location', '').strip())
            st.metric("Remote/Flexible", f"{remote_jobs}/{len(stored_jobs)}")
        
        # Market insights charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Experience distribution
            exp_levels = [job.get('experience_years', 0) for job in stored_jobs]
            exp_categories = []
            for exp in exp_levels:
                if exp <= 2:
                    exp_categories.append("Entry (0-2 yrs)")
                elif exp <= 5:
                    exp_categories.append("Mid (3-5 yrs)")
                elif exp <= 10:
                    exp_categories.append("Senior (6-10 yrs)")
                else:
                    exp_categories.append("Expert (10+ yrs)")
            
            exp_counts = {}
            for cat in exp_categories:
                exp_counts[cat] = exp_counts.get(cat, 0) + 1
            
            fig_exp = px.pie(
                values=list(exp_counts.values()),
                names=list(exp_counts.keys()),
                title="Job Distribution by Experience Level"
            )
            st.plotly_chart(fig_exp, use_container_width=True)
        
        with col2:
            # Top skills in demand
            all_skills = {}
            for job in stored_jobs:
                for skill in job.get('required_skills', []):
                    if skill and skill.strip():
                        all_skills[skill.strip()] = all_skills.get(skill.strip(), 0) + 1
            
            if all_skills:
                top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:10]
                skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Job Postings'])
                
                fig_skills = px.bar(
                    skills_df,
                    x='Job Postings',
                    y='Skill',
                    orientation='h',
                    title="Most In-Demand Skills"
                )
                st.plotly_chart(fig_skills, use_container_width=True)
        
        # Hiring trends
        st.markdown("### 📈 Hiring Trends & Opportunities")
        
        # Company hiring analysis
        company_jobs = {}
        for job in stored_jobs:
            company = job.get('company', '').strip()
            if company:
                company_jobs[company] = company_jobs.get(company, 0) + 1
        
        if company_jobs:
            top_companies = sorted(company_jobs.items(), key=lambda x: x[1], reverse=True)[:8]
            companies_df = pd.DataFrame(top_companies, columns=['Company', 'Open Positions'])
            
            col1, col2 = st.columns([2, 1])
            with col1:
                fig_companies = px.bar(
                    companies_df,
                    x='Open Positions',
                    y='Company',
                    orientation='h',
                    title="Companies with Most Openings"
                )
                st.plotly_chart(fig_companies, use_container_width=True)
            
            with col2:
                st.markdown("**Hiring Insights:**")
                for company, count in top_companies[:5]:
                    percentage = (count / len(stored_jobs)) * 100
                    st.write(f"• **{company}**: {count} jobs ({percentage:.1f}%)")
    
    def filter_jobs(self, jobs, search_query, min_exp, max_exp, companies, locations, skills, skills_match_all):
        """Filter jobs based on search criteria"""
        filtered = []
        
        for job in jobs:
            # Text search
            if search_query and search_query.strip():
                search_text = f"{job.get('title', '')} {job.get('company', '')} {job.get('raw_text', '')} {' '.join(job.get('required_skills', []))}"
                if search_query.lower() not in search_text.lower():
                    continue
            
            # Experience filter
            job_exp = job.get('experience_years', 0)
            if job_exp < min_exp or job_exp > max_exp:
                continue
            
            # Company filter
            if companies and job.get('company', '').strip() not in companies:
                continue
            
            # Location filter
            if locations and job.get('location', '').strip() not in locations:
                continue
            
            # Skills filter
            if skills:
                job_skills = [s.strip().lower() for s in job.get('required_skills', []) + job.get('preferred_skills', [])]
                selected_skills_lower = [s.lower() for s in skills]
                
                if skills_match_all:
                    if not all(skill in job_skills for skill in selected_skills_lower):
                        continue
                else:
                    if not any(skill in job_skills for skill in selected_skills_lower):
                        continue
            
            filtered.append(job)
        
        return filtered
    
    def display_job_card(self, job, index):
        """Display a job as an attractive card"""
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])
            
            with col1:
                # Job title and company
                company = job.get('company', 'Company Not Specified')
                if company and company.strip():
                    st.markdown(f"### 🏢 **{job.get('title', 'Position')}** at **{company}**")
                else:
                    st.markdown(f"### 📋 **{job.get('title', 'Position')}**")
                
                # Key details
                location = job.get('location', 'Location flexible')
                experience = job.get('experience_years', 0)
                st.markdown(f"📍 {location} • 📈 {experience} years required")
                
                # Skills preview
                required_skills = job.get('required_skills', [])[:5]
                if required_skills:
                    skills_text = ' • '.join(required_skills)
                    st.markdown(f"🛠️ **Skills:** {skills_text}")
            
            with col2:
                # Additional info
                if job.get('education_level'):
                    st.markdown(f"🎓 **Education:** {job.get('education_level', '')[:40]}...")
                
                preferred_skills = job.get('preferred_skills', [])[:3]
                if preferred_skills:
                    st.markdown(f"⭐ **Plus:** {' • '.join(preferred_skills)}")
                
                # Job posting date
                if job.get('created_at'):
                    st.markdown(f"📅 **Posted:** {job.get('created_at', '')[:10]}")
            
            with col3:
                # Actions
                st.markdown("**Actions:**")
                if st.button("👁️ View Details", key=f"view_job_{index}", use_container_width=True):
                    self.show_detailed_job_view(job)
                
                if st.button("🎯 Find Matches", key=f"match_job_{index}", use_container_width=True):
                    self.perform_job_matching(job['id'], 10, True)
                
                # Match score if available (placeholder for future enhancement)
                # st.metric("Match Score", "85%")
            
            st.divider()
    
    def show_detailed_job_view(self, job):
        """Show detailed job information in an expander"""
        with st.expander(f"📋 {job.get('title', 'Job')} - Detailed View", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**📋 Job Information**")
                st.write(f"**Title:** {job.get('title', 'N/A')}")
                st.write(f"**Company:** {job.get('company', 'Not specified')}")
                st.write(f"**Location:** {job.get('location', 'Not specified')}")
                st.write(f"**Experience Required:** {job.get('experience_years', 0)} years")
                
                if job.get('education_level'):
                    st.write(f"**Education:** {job.get('education_level')}")
                
                # Full description
                if job.get('raw_text'):
                    st.markdown("**📝 Full Description**")
                    st.text_area("", job.get('raw_text'), height=200, disabled=True)
            
            with col2:
                st.markdown("**🛠️ Required Skills**")
                required_skills = job.get('required_skills', [])
                if required_skills:
                    for skill in required_skills:
                        st.write(f"• {skill}")
                else:
                    st.write("No specific skills listed")
                
                st.markdown("**⭐ Preferred Skills**")
                preferred_skills = job.get('preferred_skills', [])
                if preferred_skills:
                    for skill in preferred_skills:
                        st.write(f"• {skill}")
                else:
                    st.write("No preferred skills listed")
                
                # Responsibilities
                responsibilities = job.get('responsibilities', [])
                if responsibilities:
                    st.markdown("**📋 Key Responsibilities**")
                    for resp in responsibilities[:5]:  # Show first 5
                        st.write(f"• {resp}")
    
    def perform_profile_job_matching(self, resume_id, jobs, threshold, max_results):
        """Perform matching between a resume profile and available jobs"""
        try:
            st.subheader("🎯 Your Job Match Results")
            
            # Get resume data
            resume_data = asyncio.run(self.resume_processor._get_resume_data(resume_id))
            if not resume_data:
                st.error("Could not load resume data")
                return
            
            # Calculate matches
            matches = []
            for job in jobs:
                # Simple matching logic based on skills and experience
                match_score = self.calculate_job_match_score(resume_data, job)
                
                if match_score >= threshold:
                    matches.append({
                        'job': job,
                        'score': match_score,
                        'match_reasons': self.get_match_reasons(resume_data, job)
                    })
            
            # Sort by match score
            matches.sort(key=lambda x: x['score'], reverse=True)
            matches = matches[:max_results]
            
            if matches:
                st.success(f"Found {len(matches)} matching opportunities!")
                
                for i, match in enumerate(matches):
                    job = match['job']
                    score = match['score']
                    reasons = match['match_reasons']
                    
                    # Match score indicator
                    score_percentage = int(score * 100)
                    if score_percentage >= 80:
                        score_color = "🟢"
                    elif score_percentage >= 60:
                        score_color = "🟡"
                    else:
                        score_color = "🟠"
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            company = job.get('company', 'Company')
                            st.markdown(f"### {score_color} **{job.get('title', 'Position')}** at **{company}**")
                            st.markdown(f"📍 {job.get('location', 'Location flexible')} • 📈 {job.get('experience_years', 0)} years required")
                            
                            # Match reasons
                            st.markdown("**Why you're a good match:**")
                            for reason in reasons[:3]:  # Show top 3 reasons
                                st.write(f"✅ {reason}")
                        
                        with col2:
                            st.metric("Match Score", f"{score_percentage}%")
                            
                            if st.button("👁️ View Job", key=f"view_match_{i}", use_container_width=True):
                                self.show_detailed_job_view(job)
                            
                            if st.button("✏️ Customize Resume", key=f"customize_match_{i}", use_container_width=True):
                                # Navigate to customizer with this job
                                st.session_state.selected_main_page = "✏️ Resume Customizer"
                                st.session_state.auto_select_job = job['id']
                                st.rerun()
                        
                        st.divider()
            else:
                st.warning(f"No jobs match your profile with the current threshold ({threshold:.1f}). Try lowering the match threshold or check if your resume has sufficient detail.")
        
        except Exception as e:
            st.error(f"Error performing job matching: {str(e)}")
            logger.error(f"Profile job matching error: {str(e)}")
    
    def calculate_job_match_score(self, resume_data, job):
        """Calculate match score between resume and job"""
        score = 0.0
        factors = 0
        
        # Skills matching
        resume_skills = set()
        if hasattr(resume_data, 'skills') and resume_data.skills:
            if hasattr(resume_data.skills, 'technical'):
                resume_skills.update([s.lower() for s in resume_data.skills.technical])
        
        job_skills = set()
        for skill in job.get('required_skills', []) + job.get('preferred_skills', []):
            if skill:
                job_skills.add(skill.lower())
        
        if job_skills:
            skills_overlap = len(resume_skills.intersection(job_skills))
            skills_score = skills_overlap / len(job_skills)
            score += skills_score * 0.6  # 60% weight for skills
            factors += 0.6
        
        # Experience matching  
        resume_exp = 0
        if hasattr(resume_data, 'experience') and resume_data.experience:
            resume_exp = getattr(resume_data.experience, 'total_years', 0)
        
        job_exp = job.get('experience_years', 0)
        if job_exp > 0:
            exp_ratio = min(resume_exp / job_exp, 1.0)  # Cap at 1.0
            score += exp_ratio * 0.3  # 30% weight for experience
            factors += 0.3
        
        # Location preference (simple check)
        job_location = job.get('location', '').lower()
        if 'remote' in job_location or not job_location.strip():
            score += 0.1  # 10% bonus for remote/flexible
            factors += 0.1
        
        return score / factors if factors > 0 else 0.0
    
    def get_match_reasons(self, resume_data, job):
        """Get reasons why this job matches the resume"""
        reasons = []
        
        # Skills match
        resume_skills = set()
        if hasattr(resume_data, 'skills') and resume_data.skills:
            if hasattr(resume_data.skills, 'technical'):
                resume_skills.update([s.lower() for s in resume_data.skills.technical])
        
        job_skills = set(skill.lower() for skill in job.get('required_skills', []) if skill)
        matching_skills = resume_skills.intersection(job_skills)
        
        if matching_skills:
            skills_list = list(matching_skills)[:3]  # Top 3
            reasons.append(f"You have {len(matching_skills)} matching skills: {', '.join(skills_list)}")
        
        # Experience match
        resume_exp = 0
        if hasattr(resume_data, 'experience') and resume_data.experience:
            resume_exp = getattr(resume_data.experience, 'total_years', 0)
        
        job_exp = job.get('experience_years', 0)
        if resume_exp >= job_exp:
            reasons.append(f"Your {resume_exp} years experience meets the {job_exp} years requirement")
        elif job_exp > 0:
            reasons.append(f"You have {resume_exp} years experience ({job_exp} years preferred)")
        
        # Location
        if 'remote' in job.get('location', '').lower():
            reasons.append("This position offers remote work flexibility")
        
        # Default reason if no specific matches
        if not reasons:
            reasons.append("Your profile shows relevant experience for this role")
        
        return reasons
    
    def resume_customizer_page(self):
        """Resume customization and cover letter generation page"""
        st.header("✏️ Resume Customizer & Cover Letter Generator")
        st.markdown("**Customize resumes and generate cover letters based on specific job requirements**")
        
        # Get stored resumes and jobs
        stored_jobs = get_stored_jobs()
        processed_resumes = get_processed_resumes()
        
        if not stored_jobs:
            st.warning("No job descriptions available. Please add jobs in the Job Management page first.")
            return
            
        if not processed_resumes:
            st.warning("No processed resumes available. Please upload resumes first.")
            return
        
        # Main tabs
        tab1, tab2, tab3 = st.tabs(["✏️ Customize Resume", "📝 Generate Cover Letter", "📋 Customization Analysis"])
        
        with tab1:
            self.resume_customization_tab(stored_jobs, processed_resumes)
        
        with tab2:
            self.cover_letter_tab(stored_jobs, processed_resumes)
            
        with tab3:
            self.customization_analysis_tab(stored_jobs, processed_resumes)

    def resume_customization_tab(self, stored_jobs, processed_resumes):
        """Resume customization interface"""
        st.subheader("✏️ Customize Resume for Specific Job")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Resume**")
            resume_options = {}
            for resume in processed_resumes:
                display_name = f"{resume.get('filename', 'Unknown')}"
                if resume.get('profile', {}).get('name'):
                    display_name = f"{resume['profile']['name']} - {resume.get('filename', 'Unknown')}"
                resume_options[display_name] = resume['id']
            
            selected_resume_name = st.selectbox("Choose resume to customize:", list(resume_options.keys()))
            selected_resume_id = resume_options[selected_resume_name] if selected_resume_name else None
        
        with col2:
            st.markdown("**Select Target Job**")
            
            # Job input options
            job_input_type = st.radio("Job Source:", ["Stored Jobs", "Job Link/Description"], horizontal=True)
            
            selected_job_id = None
            job_description_text = None
            job_title = None
            company_name = None
            
            if job_input_type == "Stored Jobs":
                job_options = {}
                for job in stored_jobs:
                    display_name = f"{job['title']} - {job['company']}"
                    job_options[display_name] = job['id']
                
                if job_options:
                    selected_job_name = st.selectbox("Choose target job:", list(job_options.keys()))
                    selected_job_id = job_options[selected_job_name] if selected_job_name else None
                else:
                    st.info("No stored jobs available. Please add jobs first or use the 'Job Link/Description' option.")
            
            else:  # Job Link/Description
                st.markdown("**Enter Job Details**")
                
                # Job link input and extraction
                job_link = st.text_input(
                    "🔗 Job Link:", 
                    placeholder="https://company.com/careers/job-id",
                    help="Paste a job posting URL to automatically extract job details"
                )
                
                # Auto-extract from URL
                if job_link and job_link.strip():
                    col_extract, col_status = st.columns([1, 2])
                    
                    with col_extract:
                        if st.button("🤖 Extract from URL", type="secondary"):
                            with st.spinner("Extracting job details from URL..."):
                                extraction_result = self.extract_job_from_url(job_link.strip())
                                
                                if extraction_result.get("success"):
                                    st.session_state.extracted_job_title = extraction_result.get("title", "")
                                    st.session_state.extracted_company = extraction_result.get("company", "")
                                    st.session_state.extracted_description = extraction_result.get("description", "")
                                    st.session_state.extracted_location = extraction_result.get("location", "")
                                    st.session_state.extracted_experience = extraction_result.get("experience", "")
                                    st.session_state.extraction_success = True
                                    
                                    # Show success with extracted details
                                    st.success("✅ Job details extracted successfully!")
                                    with st.expander("🔍 Extracted Information Preview", expanded=True):
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            if extraction_result.get("title"):
                                                st.write(f"**Job Title:** {extraction_result.get('title')}")
                                            if extraction_result.get("company"):
                                                st.write(f"**Company:** {extraction_result.get('company')}")
                                        with col2:
                                            if extraction_result.get("location"):
                                                st.write(f"**Location:** {extraction_result.get('location')}")
                                            if extraction_result.get("experience"):
                                                st.write(f"**Experience:** {extraction_result.get('experience')}")
                                        
                                        desc_preview = extraction_result.get("description", "")[:200]
                                        if desc_preview:
                                            st.write(f"**Description Preview:** {desc_preview}...")
                                    
                                    st.rerun()
                                else:
                                    st.error(f"❌ Extraction failed: {extraction_result.get('error', 'Unknown error')}")
                                    st.info("💡 **Tip:** Try a different job posting URL or copy the job description manually.")
                    
                    with col_status:
                        if hasattr(st.session_state, 'extraction_success') and st.session_state.extraction_success:
                            st.success("✅ Job details extracted and populated below")
                        else:
                            st.info("💡 Click 'Extract from URL' to automatically fill job details")
                
                # Manual job details (with auto-populated values if extracted)
                col2a, col2b = st.columns(2)
                with col2a:
                    job_title = st.text_input(
                        "📋 Job Title:", 
                        value=getattr(st.session_state, 'extracted_job_title', ''),
                        placeholder="Software Engineer"
                    )
                with col2b:
                    company_name = st.text_input(
                        "🏢 Company:", 
                        value=getattr(st.session_state, 'extracted_company', ''),
                        placeholder="Tech Corp"
                    )
                
                # Job description
                job_description_text = st.text_area(
                    "📝 Job Description:",
                    value=getattr(st.session_state, 'extracted_description', ''),
                    height=200,
                    placeholder="Paste the complete job description here...\n\nInclude requirements, responsibilities, qualifications, etc.",
                    help="Copy and paste the full job description from the job posting or extract it from the URL above"
                )
                
                # Clear extracted data button
                if hasattr(st.session_state, 'extraction_success') and st.session_state.extraction_success:
                    if st.button("🗑️ Clear Extracted Data", type="secondary"):
                        for key in ['extracted_job_title', 'extracted_company', 'extracted_description', 'extracted_location', 'extracted_experience', 'extraction_success']:
                            if hasattr(st.session_state, key):
                                delattr(st.session_state, key)
                        st.rerun()
                
                # Show URL preview if provided
                if job_link and job_link.strip():
                    with st.expander("🔗 Job Link Preview", expanded=False):
                        st.markdown(f"**URL:** {job_link}")
                        st.markdown("**Note:** This URL will be referenced in your customized resume as the source job posting")
                
                # Validate inputs
                if job_description_text and job_title:
                    selected_job_id = "custom_job"  # Flag to indicate custom job
        
        # Show validation status
        if selected_resume_id and (selected_job_id or (job_description_text and job_title)):
            if st.button("🎯 Customize Resume", type="primary"):
                with st.spinner("Customizing resume..."):
                    # Handle custom job description
                    if selected_job_id == "custom_job":
                        result = self.customize_resume_for_custom_job(
                            selected_resume_id, 
                            job_description_text or "", 
                            job_title or "Unknown Position", 
                            company_name or ""
                        )
                    else:
                        result = self.customize_resume_for_job(selected_resume_id, selected_job_id or "")
                    
                    if result and result.get('success'):
                        st.success("✅ Resume customization completed!")
                        
                        # Display customization results in improved layout
                        customized_data = result.get('customized_resume', {})
                        
                        # Create columns for content and actions
                        content_col, action_col = st.columns([3, 1])
                        
                        with content_col:
                            if customized_data.get('customized_summary'):
                                st.subheader("📝 Enhanced Summary")
                                enhanced_summary = st.text_area(
                                    "Customized Professional Summary", 
                                    value=customized_data['customized_summary'], 
                                    height=120,
                                    key="customized_summary_edit",
                                    help="You can edit this summary before copying or downloading"
                                )
                            
                            if customized_data.get('emphasized_skills'):
                                st.subheader("🔍 Emphasized Skills")
                                skills_text = " • ".join(customized_data['emphasized_skills'])
                                st.success(f"**Prioritized Skills:** {skills_text}")
                            
                            if customized_data.get('keyword_suggestions'):
                                st.subheader("🏷️ Recommended Keywords")
                                keywords_text = ", ".join(customized_data['keyword_suggestions'])
                                st.write(keywords_text)
                                
                                # Make keywords copyable
                                st.code(keywords_text, language="text")
                            
                            if customized_data.get('customization_summary'):
                                st.subheader("📋 Customization Summary")
                                st.info(customized_data['customization_summary'])
                        
                        with action_col:
                            st.markdown("**📥 Download Options**")
                            
                            # Prepare full customized resume text
                            job_info = job_title if job_title else result.get('job_title', 'Position')
                            company_info = company_name if company_name else result.get('company', 'Company')
                            
                            full_resume_text = f"""
CUSTOMIZED RESUME FOR: {job_info} at {company_info}

ENHANCED PROFESSIONAL SUMMARY:
{customized_data.get('customized_summary', 'N/A')}

EMPHASIZED SKILLS:
{' • '.join(customized_data.get('emphasized_skills', []))}

RECOMMENDED KEYWORDS:
{', '.join(customized_data.get('keyword_suggestions', []))}

CUSTOMIZATION SUMMARY:
{customized_data.get('customization_summary', 'N/A')}

Original Resume: {selected_resume_name}
Target Position: {job_info}
Company: {company_info}
Generated: {result.get('generated_at', 'Unknown')}
                            """.strip()
                            
                            # Download as TXT
                            if st.download_button(
                                label="📄 Download TXT",
                                data=full_resume_text,
                                file_name=f"customized_resume_{job_info.lower().replace(' ', '_')}.txt",
                                mime="text/plain"
                            ):
                                st.success("✅ Resume downloaded!")
                            
                            # Download as HTML
                            html_resume = f"""
                            <html>
                            <head>
                                <title>Customized Resume - {job_info}</title>
                                <style>
                                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                                    h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; }}
                                    h2 {{ color: #34495e; margin-top: 30px; }}
                                    .summary {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #3498db; }}
                                    .skills {{ background: #e8f5e8; padding: 10px; border-radius: 5px; }}
                                    .keywords {{ background: #fff3cd; padding: 10px; border-radius: 5px; }}
                                </style>
                            </head>
                            <body>
                                <h1>Customized Resume</h1>
                                <p><strong>Position:</strong> {job_info}<br>
                                <strong>Company:</strong> {company_info}<br>
                                <strong>Original Resume:</strong> {selected_resume_name}</p>
                                
                                <h2>Enhanced Professional Summary</h2>
                                <div class="summary">{customized_data.get('customized_summary', 'N/A')}</div>
                                
                                <h2>Emphasized Skills</h2>
                                <div class="skills">{' • '.join(customized_data.get('emphasized_skills', []))}</div>
                                
                                <h2>Recommended Keywords</h2>
                                <div class="keywords">{', '.join(customized_data.get('keyword_suggestions', []))}</div>
                                
                                <h2>Customization Summary</h2>
                                <p>{customized_data.get('customization_summary', 'N/A')}</p>
                            </body>
                            </html>
                            """
                            
                            if st.download_button(
                                label="🌐 Download HTML",
                                data=html_resume,
                                file_name=f"customized_resume_{job_info.lower().replace(' ', '_')}.html",
                                mime="text/html"
                            ):
                                st.success("✅ HTML downloaded!")
                            
                            # Download as PDF
                            try:
                                # Prepare resume data for PDF generation
                                resume_pdf_data = {
                                    'profile': {
                                        'name': selected_resume_name,
                                        'phone': '',  # Would need to extract from original resume
                                        'email': '',  # Would need to extract from original resume
                                        'linkedin': '',
                                        'location': ''
                                    },
                                    'summary': customized_data.get('customized_summary', ''),
                                    'skills': customized_data.get('emphasized_skills', []),
                                    'experience': [],  # Would need original resume experience
                                    'education': [],   # Would need original resume education
                                    'projects': []     # Would need original resume projects
                                }
                                
                                pdf_bytes = generate_resume_pdf(resume_pdf_data)
                                
                                if st.download_button(
                                    label="📄 Download PDF",
                                    data=pdf_bytes,
                                    file_name=f"customized_resume_{job_info.lower().replace(' ', '_')}.pdf",
                                    mime="application/pdf"
                                ):
                                    st.success("✅ PDF downloaded!")
                            except Exception as e:
                                st.warning(f"PDF generation temporarily unavailable: {str(e)}")
                                logger.error(f"PDF generation error: {str(e)}")
                            
                            # Copy button
                            if st.button("📋 Copy All", help="Display text for easy copying"):
                                st.code(full_resume_text, language="text")
                                st.success("✅ Text displayed above - select and copy!")
                    else:
                        st.error(f"❌ Failed to customize resume: {result.get('error', 'Unknown error')}")
        elif selected_resume_id:
            if job_input_type == "Job Link/Description":
                if not job_description_text:
                    st.warning("⚠️ Please enter the job description")
                elif not job_title:
                    st.warning("⚠️ Please enter the job title")
            else:
                st.info("💡 Select a target job to continue")

    def cover_letter_tab(self, stored_jobs, processed_resumes):
        """Cover letter generation interface"""
        st.subheader("📝 Generate Personalized Cover Letter")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Candidate Resume**")
            resume_options = {}
            for resume in processed_resumes:
                display_name = f"{resume.get('filename', 'Unknown')}"
                if resume.get('profile', {}).get('name'):
                    display_name = f"{resume['profile']['name']} - {resume.get('filename', 'Unknown')}"
                resume_options[display_name] = resume['id']
            
            selected_resume_name = st.selectbox("Choose candidate:", list(resume_options.keys()), key="cover_resume")
            selected_resume_id = resume_options[selected_resume_name] if selected_resume_name else None
        
        with col2:
            st.markdown("**Select Job Input Method**")
            job_input_type_cover = st.radio(
                "Choose input method:",
                options=["Stored Jobs", "Job Link/Description"],
                key="cover_job_input_type",
                help="Select from stored jobs or paste a job description"
            )
            
            if job_input_type_cover == "Stored Jobs":
                job_options = {}
                for job in stored_jobs:
                    display_name = f"{job['title']} - {job['company']}"
                    job_options[display_name] = job['id']
                
                selected_job_name = st.selectbox("Choose position:", list(job_options.keys()), key="cover_job")
                selected_job_id = job_options[selected_job_name] if selected_job_name else None
                job_description_text_cover = None
                job_title_cover = None
                company_name_cover = None
            else:
                # Custom job description input
                selected_job_id = "custom_job"
                
                job_title_cover = st.text_input(
                    "Job Title (Optional):",
                    placeholder="e.g., Senior Software Engineer",
                    key="cover_job_title"
                )
                
                company_name_cover = st.text_input(
                    "Company Name (Optional):",
                    placeholder="e.g., Tech Corp",
                    key="cover_company_name"
                )
                
                job_description_text_cover = st.text_area(
                    "Job Description/Link:",
                    placeholder="Paste the job description or link here...",
                    height=150,
                    key="cover_job_description"
                )
        
        # Generate cover letter button and logic
        if selected_resume_id:
            if job_input_type_cover == "Stored Jobs" and selected_job_id:
                if st.button("📝 Generate Cover Letter", type="primary", key="generate_cover_stored"):
                    with st.spinner("Generating personalized cover letter..."):
                        result = self.generate_cover_letter_for_job(selected_resume_id, selected_job_id)
                        self._display_cover_letter_result(result)
            elif job_input_type_cover == "Job Link/Description" and job_description_text_cover:
                if st.button("📝 Generate Cover Letter", type="primary", key="generate_cover_custom"):
                    with st.spinner("Generating personalized cover letter..."):
                        result = self.generate_cover_letter_for_custom_job(
                            selected_resume_id,
                            job_description_text_cover or "",
                            job_title_cover or "Unknown Position",
                            company_name_cover or ""
                        )
                        self._display_cover_letter_result(result)
            elif job_input_type_cover == "Job Link/Description" and not job_description_text_cover:
                st.warning("⚠️ Please enter the job description")
    
    def _display_cover_letter_result(self, result):
        """Display cover letter generation result"""
        if result and result.get('success'):
            st.success("✅ Cover letter generated successfully!")
            
            # Display cover letter
            cover_letter = result.get('cover_letter', '') or ""
            
            st.subheader(f"📝 Cover Letter for {result.get('job_title', 'Position')}")
            
            # Create two columns for better layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Display cover letter in an easy-to-copy format
                st.markdown("**Generated Cover Letter:**")
                
                # Use code block for easy copying
                st.code(cover_letter, language="text")
                
                # Also provide a text area for editing
                edited_cover_letter = st.text_area(
                    "Edit Cover Letter (Optional)", 
                    value=cover_letter, 
                    height=300,
                    help="You can edit the cover letter before copying or downloading",
                    key="edit_cover_letter"
                ) or cover_letter
            
            with col2:
                st.markdown("**Actions:**")
                
                # Copy to clipboard button (using JavaScript)
                if st.button("📋 Copy to Clipboard", help="Copy the cover letter to your clipboard", key="copy_cover"):
                    # Use Streamlit's built-in copy functionality
                    st.write("📋 **Copy this text:**")
                    st.code(edited_cover_letter, language="text")
                    st.success("✅ Text displayed above for copying!")
                
                # Download option
                if st.download_button(
                    label="📄 Download as TXT",
                    data=edited_cover_letter or "",
                    file_name=f"cover_letter_{result.get('candidate_name', 'candidate')}_{result.get('company', 'company')}.txt",
                    mime="text/plain",
                    key="download_cover_txt"
                ):
                    st.success("Cover letter downloaded!")
                
                # Simple HTML download for better formatting
                html_content = f"""
                <html>
                <head><title>Cover Letter</title></head>
                <body>
                <h2>Cover Letter for {result.get('job_title', 'Position')}</h2>
                <pre style="font-family: Arial; white-space: pre-wrap;">{edited_cover_letter}</pre>
                </body>
                </html>
                """
                if st.download_button(
                    label="📄 Download as HTML",
                    data=html_content,
                    file_name=f"cover_letter_{result.get('candidate_name', 'candidate')}_{result.get('company', 'company')}.html",
                    mime="text/html",
                    key="download_cover_html"
                ):
                    st.success("Cover letter downloaded as HTML!")
            
            # Additional formatting options
            with st.expander("📋 Formatted for Email", expanded=False):
                st.markdown("**Email-ready format:**")
                email_format = cover_letter.replace('\n\n', '\n').replace('\n', '<br>')
                st.markdown(email_format, unsafe_allow_html=True)
                
            # Character and word count
            safe_text = edited_cover_letter or ""
            char_count = len(safe_text)
            word_count = len(safe_text.split()) if safe_text else 0
            st.caption(f"📊 Character count: {char_count} | Word count: {word_count}")
        else:
            st.error(f"❌ Failed to generate cover letter: {result.get('error', 'Unknown error')}")

    def customization_analysis_tab(self, stored_jobs, processed_resumes):
        """Customization analysis and suggestions interface"""
        st.subheader("📋 Resume Customization Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Select Resume to Analyze**")
            resume_options = {}
            for resume in processed_resumes:
                display_name = f"{resume.get('filename', 'Unknown')}"
                if resume.get('profile', {}).get('name'):
                    display_name = f"{resume['profile']['name']} - {resume.get('filename', 'Unknown')}"
                resume_options[display_name] = resume['id']
            
            selected_resume_name = st.selectbox("Choose resume:", list(resume_options.keys()), key="analysis_resume")
            selected_resume_id = resume_options[selected_resume_name] if selected_resume_name else None
        
        with col2:
            st.markdown("**Select Target Job**")
            job_options = {}
            for job in stored_jobs:
                display_name = f"{job['title']} - {job['company']}"
                job_options[display_name] = job['id']
            
            selected_job_name = st.selectbox("Choose job:", list(job_options.keys()), key="analysis_job")
            selected_job_id = job_options[selected_job_name] if selected_job_name else None
        
        if selected_resume_id and selected_job_id:
            if st.button("🔍 Analyze Customization Needs", type="primary"):
                with st.spinner("Analyzing customization requirements..."):
                    result = self.analyze_customization_needs(selected_resume_id, selected_job_id)
                    if result and result.get('success'):
                        st.success("✅ Analysis completed!")
                        
                        suggestions = result.get('suggestions', {})
                        
                        # Display skill gaps
                        if suggestions.get('skill_gaps'):
                            st.subheader("⚠️ Skill Gaps")
                            for gap in suggestions['skill_gaps']:
                                st.error(f"Missing: {gap}")
                        
                        # Display experience recommendations
                        if suggestions.get('experience_recommendations'):
                            st.subheader("💼 Experience Recommendations")
                            for rec in suggestions['experience_recommendations']:
                                st.info(rec)
                        
                        # Display keyword suggestions
                        if suggestions.get('keyword_suggestions'):
                            st.subheader("🏷️ Keyword Suggestions")
                            cols = st.columns(3)
                            for i, keyword in enumerate(suggestions['keyword_suggestions']):
                                with cols[i % 3]:
                                    st.success(keyword)
                        
                        # Display priority changes
                        if suggestions.get('priority_changes'):
                            st.subheader("🎯 Priority Changes")
                            for change in suggestions['priority_changes']:
                                priority = change.get('priority', 'medium')
                                if priority == 'high':
                                    st.error(f"🔴 HIGH: {change.get('change', '')}")
                                elif priority == 'medium':
                                    st.warning(f"🟡 MEDIUM: {change.get('change', '')}")
                                else:
                                    st.info(f"🟢 LOW: {change.get('change', '')}")
                        
                        # Overall assessment
                        if suggestions.get('overall_assessment'):
                            st.subheader("📊 Overall Assessment")
                            st.markdown(suggestions['overall_assessment'])
                    else:
                        st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    def customize_resume_for_job(self, resume_id: str, job_id: str):
        """Customize resume for specific job"""
        try:
            # Get resume and job data
            resume_data = asyncio.run(self.resume_processor._get_resume_data(resume_id))
            job_data = asyncio.run(self.job_processor.get_job_data(job_id))
            
            if not resume_data or not job_data:
                return {"success": False, "error": "Failed to retrieve resume or job data"}
            
            # Call customization service
            result = asyncio.run(resume_customizer.customize_resume(resume_data, job_data))
            return result
            
        except Exception as e:
            logger.error(f"Error customizing resume: {str(e)}")
            return {"success": False, "error": str(e)}

    def customize_resume_for_custom_job(self, resume_id: str, job_description: str, job_title: str, company_name: str = ""):
        """Customize resume for custom job description"""
        try:
            # Get resume data
            resume_data = asyncio.run(self.resume_processor._get_resume_data(resume_id))
            
            if not resume_data:
                return {"success": False, "error": "Failed to retrieve resume data"}
            
            # Create temporary job data object
            from app.models.resume_data import JobDescription
            from datetime import datetime
            
            temp_job_data = JobDescription(
                title=job_title,
                company=company_name or "Target Company",
                raw_text=job_description,
                created_at=datetime.now()
            )
            
            # Parse job description to extract requirements
            try:
                parsed_job = asyncio.run(self.resume_processor.process_job_description(job_description))
                if parsed_job:
                    temp_job_data.required_skills = parsed_job.required_skills
                    temp_job_data.preferred_skills = parsed_job.preferred_skills
                    temp_job_data.experience_years = parsed_job.experience_years
                    temp_job_data.responsibilities = parsed_job.responsibilities
            except Exception as parse_error:
                logger.warning(f"Failed to parse custom job description: {parse_error}")
                # Continue with basic job data
            
            # Call customization service
            result = asyncio.run(resume_customizer.customize_resume(resume_data, temp_job_data))
            
            # Add custom job info to result
            if result and result.get('success'):
                result['job_title'] = job_title
                result['company'] = company_name or "Target Company"
                result['generated_at'] = datetime.now().isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Error customizing resume for custom job: {str(e)}")
            return {"success": False, "error": str(e)}

    def generate_cover_letter_for_job(self, resume_id: str, job_id: str):
        """Generate cover letter for specific job"""
        try:
            # Get resume and job data
            resume_data = asyncio.run(self.resume_processor._get_resume_data(resume_id))
            job_data = asyncio.run(self.job_processor.get_job_data(job_id))
            
            if not resume_data or not job_data:
                return {"success": False, "error": "Failed to retrieve resume or job data"}
            
            # Call cover letter generation service
            result = asyncio.run(resume_customizer.generate_cover_letter(resume_data, job_data))
            return result
            
        except Exception as e:
            logger.error(f"Error generating cover letter: {str(e)}")
            return {"success": False, "error": str(e)}

    def generate_cover_letter_for_custom_job(self, resume_id: str, job_description: str, job_title: str, company_name: str):
        """Generate cover letter for custom job description"""
        try:
            # Get resume data
            resume_data = asyncio.run(self.resume_processor._get_resume_data(resume_id))
            
            if not resume_data:
                return {"success": False, "error": "Failed to retrieve resume data"}
            
            # Create a JobDescription object
            job_data = JobDescription(
                title=job_title,
                company=company_name,
                raw_text=job_description,
                summary=job_description[:500] + "..." if len(job_description) > 500 else job_description
            )
            
            # Call cover letter generation service
            result = asyncio.run(resume_customizer.generate_cover_letter(resume_data, job_data))
            return result
            
        except Exception as e:
            logger.error(f"Error generating cover letter for custom job: {str(e)}")
            return {"success": False, "error": str(e)}

    def analyze_customization_needs(self, resume_id: str, job_id: str):
        """Analyze customization needs"""
        try:
            # Get resume and job data
            resume_data = asyncio.run(self.resume_processor._get_resume_data(resume_id))
            job_data = asyncio.run(self.job_processor.get_job_data(job_id))
            
            if not resume_data or not job_data:
                return {"success": False, "error": "Failed to retrieve resume or job data"}
            
            # Call analysis service
            result = asyncio.run(resume_customizer.get_customization_suggestions(resume_data, job_data))
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing customization needs: {str(e)}")
            return {"success": False, "error": str(e)}

    def analytics_page(self):
        """Enhanced analytics and insights page focused on job market intelligence"""
        st.header("📊 Job Market Analytics & Candidate Insights")
        st.markdown("**Understand the job market landscape and identify opportunities for skill development**")
        
        try:
            # Get analytics data
            analytics_data = self.get_analytics_data()
            
            if not analytics_data:
                st.info("📊 No data available for analytics. Add some jobs and process resumes first.")
                
                # Show what data is available
                jobs_count = len(get_stored_jobs())
                resumes_count = len(get_processed_resumes())
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Jobs Available", jobs_count)
                    if jobs_count == 0:
                        if st.button("➕ Add Some Jobs"):
                            st.session_state.selected_main_page = "📋 Job Management"
                            st.rerun()
                
                with col2:
                    st.metric("Resumes Processed", resumes_count)
                    if resumes_count == 0:
                        if st.button("📄 Upload Resumes"):
                            st.session_state.selected_main_page = "📄 Resume Upload"
                            st.rerun()
                
                return
            
            # Create tabs for different types of analytics
            tab1, tab2, tab3, tab4 = st.tabs([
                "🎯 Job Market Overview", 
                "🛠️ Skills Intelligence", 
                "📈 Career Insights", 
                "🏢 Company Analysis"
            ])
            
            with tab1:
                try:
                    self.display_job_market_overview(analytics_data)
                except Exception as e:
                    st.error(f"Error loading job market overview: {str(e)}")
                    logger.error(f"Job market overview error: {str(e)}")
            
            with tab2:
                try:
                    self.display_skills_intelligence(analytics_data)
                except Exception as e:
                    st.error(f"Error loading skills intelligence: {str(e)}")
                    logger.error(f"Skills intelligence error: {str(e)}")
            
            with tab3:
                try:
                    self.display_career_insights(analytics_data)
                except Exception as e:
                    st.error(f"Error loading career insights: {str(e)}")
                    logger.error(f"Career insights error: {str(e)}")
            
            with tab4:
                try:
                    self.display_company_analysis(analytics_data)
                except Exception as e:
                    st.error(f"Error loading company analysis: {str(e)}")
                    logger.error(f"Company analysis error: {str(e)}")
                
        except Exception as e:
            st.error(f"Error loading analytics data: {str(e)}")
            logger.error(f"Analytics page error: {str(e)}")
    
    
    def display_resume_json_preview(self, resume_data, filename: str):
        """Display parsed resume data as JSON with completeness indicators"""
        st.subheader(f"📋 Parsed Data: {filename}")
        
        # Calculate completeness
        required_fields = {
            "profile.name": "Name",
            "profile.email": "Email",
            "profile.phone": "Phone",
            "experience.total_years": "Experience Years",
            "skills.technical": "Technical Skills"
        }
        
        completeness_results = {}
        for field_path, field_name in required_fields.items():
            parts = field_path.split('.')
            current = resume_data
            
            try:
                for part in parts:
                    current = current.get(part) if isinstance(current, dict) else getattr(current, part, None)
                
                has_value = current is not None and current != "" and current != []
                completeness_results[field_name] = has_value
            except (AttributeError, KeyError):
                completeness_results[field_name] = False
        
        # Display completeness score
        completeness_score = sum(completeness_results.values()) / len(completeness_results)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.metric("Completeness Score", f"{completeness_score:.0%}")
        
        with col2:
            if completeness_score >= 0.8:
                st.success("✅ Excellent")
            elif completeness_score >= 0.6:
                st.warning("⚠️ Good")
            else:
                st.error("❌ Needs Review")
        
        # Display field status
        st.markdown("**Field Status:**")
        cols = st.columns(5)
        for i, (field_name, has_value) in enumerate(completeness_results.items()):
            with cols[i % 5]:
                if has_value:
                    st.success(f"✅ {field_name}")
                else:
                    st.error(f"❌ {field_name}")
        
        # Convert resume data to dict for JSON display
        resume_dict = {
            "profile": resume_data.profile.__dict__ if resume_data.profile else {},
            "experience": resume_data.experience.__dict__ if resume_data.experience else {},
            "skills": resume_data.skills.__dict__ if resume_data.skills else {},
            "topics": resume_data.topics.__dict__ if resume_data.topics else {},
            "tools_libraries": resume_data.tools_libraries.__dict__ if resume_data.tools_libraries else {},
            "summary": resume_data.summary,
            "key_strengths": resume_data.key_strengths
        }
        
        # Tabbed view for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["👤 Profile", "💼 Experience & Skills", "📊 Full JSON", "🔍 Raw Text"])
        
        with tab1:
            st.json(resume_dict.get("profile", {}))
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Experience:**")
                st.json(resume_dict.get("experience", {}))
            with col2:
                st.markdown("**Skills:**")
                st.json(resume_dict.get("skills", {}))
        
        with tab3:
            st.json(resume_dict)
        
        with tab4:
            st.text_area("Raw Resume Text", resume_data.raw_text, height=300, disabled=True)
        
        # Add feedback mechanism
        st.markdown("---")
        st.markdown("**Was this parsing accurate?**")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Accurate", key=f"accurate_{resume_data.id}"):
                st.success("Thank you for your feedback!")
        with col2:
            if st.button("👎 Needs Improvement", key=f"inaccurate_{resume_data.id}"):
                st.info("We'll work on improving our parsing. You can edit the resume data in the customizer.")
        with col3:
            if st.button("🔄 Reprocess", key=f"reprocess_{resume_data.id}"):
                st.info("Reprocessing feature coming soon!")
    
    def process_uploaded_resumes(self, uploaded_files, progress_bar, status_text, batch_process, show_detailed_errors=True):
        """Process uploaded resume files with improved error handling"""
        try:
            total_files = len(uploaded_files)
            processed_count = 0
            error_count = 0
            
            st.info(f"🔄 Starting processing of {total_files} files...")
            
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                progress_bar.progress((i + 1) / total_files)
                
                # Save uploaded file temporarily
                temp_path = f"data/temp/{uploaded_file.name}"
                Path("data/temp").mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process resume
                try:
                    resume_data = asyncio.run(
                        self.resume_processor.process_resume_file(temp_path, uploaded_file.name)
                    )
                    
                    if resume_data:
                        st.success(f"✅ Successfully processed {uploaded_file.name}")
                        
                        # Safe access to profile data for expander title
                        candidate_name = 'Unknown Candidate'
                        candidate_title = 'No Title'
                        
                        if resume_data.profile:
                            candidate_name = getattr(resume_data.profile, 'name', '') or 'Unknown Candidate'
                            candidate_title = getattr(resume_data.profile, 'title', '') or 'No Title'
                        
                        # Display resume summary with JSON preview
                        with st.expander(f"📄 {candidate_name} - {candidate_title}", expanded=True):
                            if resume_data:
                                self.display_resume_json_preview(resume_data, uploaded_file.name)
                            else:
                                st.error("Resume data is empty or invalid")
                        
                        processed_count += 1
                    else:
                        st.error(f"❌ Processing returned empty data for {uploaded_file.name}")
                        error_count += 1
                    
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                    logger.error(f"Resume processing error for {uploaded_file.name}: {str(e)}")
                    
                    error_count += 1
                    
                    # Show more detailed error information if requested
                    if show_detailed_errors:
                        with st.expander("🔍 Error Details", expanded=False):
                            st.write("**Error Type:**", type(e).__name__)
                            st.write("**Error Message:**", str(e))
                            st.write("**File:**", uploaded_file.name)
                            st.write("**File Size:**", f"{len(uploaded_file.getbuffer())} bytes")
                            
                            # Additional debugging information
                            try:
                                # Try to read first few lines of the file
                                content_preview = uploaded_file.read(500).decode('utf-8', errors='ignore')
                                uploaded_file.seek(0)  # Reset file pointer
                                st.write("**File Content Preview:**")
                                st.code(content_preview[:200] + "..." if len(content_preview) > 200 else content_preview)
                            except:
                                st.write("**File Content:** Unable to preview file content")
                            
                            st.write("**Troubleshooting Suggestions:**")
                            st.write("• Check if file contains readable text")
                            st.write("• Ensure file is not corrupted or password-protected")
                            st.write("• Try converting to PDF or TXT format")
                            st.write("• Verify the file is a valid resume document")
                            st.write("• Check if API keys are properly configured")
                
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
            
            # Show final summary
            progress_bar.progress(1.0)
            if error_count == 0:
                status_text.success(f"✅ All {total_files} files processed successfully!")
                st.balloons()
                
                # Suggest next actions
                st.success("🎉 **What's next?**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("✏️ Customize Resumes", type="primary", use_container_width=True):
                        st.session_state.selected_main_page = "✏️ Resume Customizer"
                        st.rerun()
                with col2:
                    if st.button("🎯 Find Job Matches", use_container_width=True):
                        st.session_state.selected_main_page = "🎯 Job Matching"
                        st.rerun()
                with col3:
                    if st.button("🔍 Search Candidates", use_container_width=True):
                        st.session_state.selected_main_page = "🔍 Search Candidates"
                        st.rerun()
            else:
                status_text.warning(f"⚠️ Processing complete: {processed_count} successful, {error_count} failed")
                
                if processed_count > 0:
                    st.info("💡 **Some resumes were processed successfully!** You can now use the Resume Customizer with the successfully processed resumes.")
            
            # Clear cache to show updated data
            st.cache_data.clear()
            
        except Exception as e:
            st.error(f"Fatal error during processing: {str(e)}")
            logger.error(f"Fatal processing error: {str(e)}")
            
            with st.expander("🚨 System Error Details", expanded=True):
                st.write("A system-level error occurred during processing.")
                st.write("**Error:**", str(e))
                st.write("**Possible causes:**")
                st.write("• System resources exhausted")
                st.write("• API service unavailable") 
                st.write("• Database connection issues")
                st.write("• File system permissions")
    
    def job_management_page(self):
        """Job description management page with clean UI"""
        st.header("📋 Job Management")
        
        st.info("💡 Use the sidebar to add new jobs.")
        
        self.display_jobs_table()
    
    # add_job_form removed in favor of sidebar_add_job

    def display_jobs_table(self):
        """Display jobs in a clean table format"""
        try:
            jobs = get_stored_jobs()
            
            if not jobs:
                st.info("📝 No jobs stored yet. Add your first job using the form above!")
                
                # Quick action buttons
                col1, col2, col3 = st.columns(3)
                with col2:
                    if st.button("📊 Load Sample Jobs", use_container_width=True):
                        with st.spinner("Loading sample jobs..."):
                            try:
                                result = asyncio.run(self.data_pipeline.process_sample_data())
                                st.success(f"✅ Loaded {result['jobs']['processed']} sample jobs!")
                                st.cache_data.clear()
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to load sample jobs: {e}")
                return
            
            # Summary metrics
            st.subheader(f"📊 Jobs Overview ({len(jobs)} total)")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_experience = sum(job.get('experience_years', 0) for job in jobs) / len(jobs)
                st.metric("Avg Experience", f"{avg_experience:.1f} yrs")
            
            with col2:
                companies = set(job.get('company', '') for job in jobs if job.get('company'))
                st.metric("Companies", len(companies))
            
            with col3:
                total_skills = sum(len(job.get('required_skills', [])) for job in jobs)
                st.metric("Total Skills", total_skills)
            
            with col4:
                locations = set(job.get('location') for job in jobs if job.get('location'))
                st.metric("Locations", len(locations))
            
            # Enhanced Jobs Display
            st.subheader("� Available Positions")
            
            # Display jobs as cards for better visual appeal
            for i, job in enumerate(jobs):
                with st.container():
                    # Create a card-like display
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        # Prominent job title and company
                        company_name = job.get('company', 'Company Not Specified')
                        if company_name and company_name.strip():
                            st.markdown(f"### 🏢 **{job.get('title', 'Position')}** at **{company_name}**")
                        else:
                            st.markdown(f"### 📋 **{job.get('title', 'Position')}**")
                        
                        # Location and experience
                        location = job.get('location', 'Location flexible')
                        experience = job.get('experience_years', 0)
                        st.markdown(f"📍 {location} • 📈 {experience} years experience")
                        
                        # Key skills preview
                        required_skills = job.get('required_skills', [])
                        if required_skills:
                            skills_display = ', '.join(required_skills[:4])
                            if len(required_skills) > 4:
                                skills_display += f" +{len(required_skills) - 4} more"
                            st.markdown(f"🛠️ **Key Skills:** {skills_display}")
                    
                    with col2:
                        # Additional job info
                        if job.get('education_level'):
                            st.markdown(f"🎓 **Education:** {job.get('education_level', 'Not specified')[:50]}...")
                        
                        # Preferred skills
                        preferred_skills = job.get('preferred_skills', [])
                        if preferred_skills:
                            pref_display = ', '.join(preferred_skills[:3])
                            if len(preferred_skills) > 3:
                                pref_display += f" +{len(preferred_skills) - 3} more"
                            st.markdown(f"⭐ **Preferred:** {pref_display}")
                    
                    with col3:
                        # Action buttons
                        st.markdown("**Actions:**")
                        if st.button("🔍 Find Matches", key=f"find_{i}", use_container_width=True):
                            self.perform_job_matching(job['id'], 10, True)
                        
                        if st.button("👁️ Details", key=f"view_{i}", use_container_width=True):
                            self.show_job_details(job)
                    
                    st.divider()
            
            # Quick stats table for overview
            st.subheader("📊 Quick Overview")
            jobs_data = []
            for job in jobs:
                jobs_data.append({
                    "🏢 Company": job.get('company', 'Not specified') or 'Not specified',
                    "📋 Role": job.get('title', 'N/A'),
                    "📍 Location": job.get('location', 'Flexible') or 'Flexible',
                    "📈 Experience": f"{job.get('experience_years', 0)} yrs",
                    "🛠️ Skills Count": len(job.get('required_skills', [])),
                    "📅 Added": job.get('created_at', '')[:10] if job.get('created_at') else 'N/A'
                })
            
            jobs_df = pd.DataFrame(jobs_data)
            st.dataframe(jobs_df, use_container_width=True, hide_index=True)
            
            # Job selection for actions
            st.divider()
            st.subheader("🎯 Job Actions")
            
            job_options = {f"{job['title']} - {job['company']}": job for job in jobs}
            selected_job_name = st.selectbox(
                "Select a job for actions:",
                options=list(job_options.keys()),
                help="Choose a job to perform actions on"
            )
            
            if selected_job_name:
                selected_job = job_options[selected_job_name]
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    if st.button("🔍 Find Candidates", use_container_width=True):
                        self.perform_job_matching(selected_job['id'], 10, True)
                
                with col2:
                    if st.button("👁️ View Details", use_container_width=True):
                        self.show_job_details(selected_job)
                
                with col3:
                    if st.button("✏️ Edit Job", use_container_width=True):
                        st.info("🚧 Edit functionality coming soon!")
                
                with col4:
                    if st.button("🗑️ Delete Job", use_container_width=True, type="secondary"):
                        if st.session_state.get('confirm_delete'):
                            # Delete confirmation logic would go here
                            st.warning("Delete confirmation would be implemented here")
                        else:
                            st.session_state.confirm_delete = True
                            st.warning("Click again to confirm deletion")
                
        except Exception as e:
            st.error(f"Error loading jobs: {str(e)}")
            logger.error(f"Jobs display error: {str(e)}")

    def show_job_details(self, job):
        """Show detailed job information"""
        with st.expander(f"📋 {job['title']} - {job['company']}", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("**Job Information**")
                st.write(f"**Title:** {job['title']}")
                st.write(f"**Company:** {job['company']}")
                st.write(f"**Location:** {job.get('location', 'Not specified')}")
                st.write(f"**Experience Required:** {job.get('experience_years', 'N/A')} years")
                
                if job.get('summary'):
                    st.markdown("**AI Summary**")
                    st.write(job['summary'])
            
            with col2:
                st.markdown("**Metadata**")
                st.write(f"**Job ID:** {job['id'][:8]}...")
                st.write(f"**Created:** {job.get('created_at', 'N/A')[:16]}")
                
                # Required skills
                skills = job.get('required_skills', [])
                if skills:
                    st.markdown(f"**Required Skills ({len(skills)})**")
                    for skill in skills[:10]:  # Show first 10 skills
                        st.write(f"• {skill}")
                    if len(skills) > 10:
                        st.write(f"... and {len(skills) - 10} more")
            
            # Job description
            if job.get('raw_text'):
                st.markdown("**Full Job Description**")
                st.text_area("", job['raw_text'], height=200, disabled=True)

    def save_job_description(self, title, company, description, experience_years, location, source_url=""):
        """Save job description to vector store"""
        try:
            job_data = asyncio.run(
                self.job_processor.process_and_store_job(
                    job_text=description,
                    title=title,
                    company=company,
                    experience_years=experience_years,
                    location=location
                )
            )
            
            # If we have a source URL, add it to the job data
            if source_url and source_url.strip():
                # Add source URL to job metadata (this would require job_processor enhancement)
                st.info(f"📎 Source: {source_url}")
            
            st.success(f"✅ Successfully saved job: {title}")
            
            # Display job summary
            with st.expander(f"📋 {title} - {company}"):
                self.display_job_summary(job_data)
                if source_url and source_url.strip():
                    st.markdown(f"**🔗 Source URL:** {source_url}")
            
        except Exception as e:
            st.error(f"❌ Failed to save job: {str(e)}")
    
    def perform_job_matching(self, job_id, top_k, show_details=True):
        """Perform job matching using enhanced job matching interface"""
        try:
            # Get job data
            job_data = asyncio.run(self.job_processor.get_job_data(job_id))
            if not job_data:
                st.error("Job not found")
                return
            
            st.info(f"🔍 Finding matches for: {job_data.title}")
            
            # Use the job processor's find_candidates_for_job method
            with st.spinner("Searching for candidates..."):
                candidates = asyncio.run(self.job_processor.find_candidates_for_job(job_id, top_k))
            
            if not candidates:
                st.warning("No matching candidates found")
                return
            
            # Display results
            st.subheader(f"🎯 Top {len(candidates)} Candidates")
            
            # Display detailed results
            if show_details:
                self.display_vector_search_results(candidates)
            
        except Exception as e:
            st.error(f"Error performing matching: {str(e)}")
            logger.error(f"Job matching error: {str(e)}")
    
    def display_vector_search_results(self, candidates, show_details=True):
        """Display vector search results in a user-friendly format"""
        try:
            if not candidates:
                st.warning("No candidates found")
                return
            
            # Create results dataframe
            results_data = []
            for i, candidate in enumerate(candidates, 1):
                similarity = candidate.get('similarity', 0)
                metadata = candidate.get('metadata', {})
                
                # Extract candidate info
                candidate_info = {
                    'Rank': i,
                    'Similarity': f"{similarity:.3f}",
                    'Name': metadata.get('name', 'N/A'),
                    'Title': metadata.get('title', 'N/A'),
                    'Experience': f"{metadata.get('experience_years', 0)} years",
                    'Skills Count': len(metadata.get('skills', [])),
                    'ID': candidate.get('candidate_id', 'Unknown')[:12] + "..."
                }
                results_data.append(candidate_info)
            
            # Display results table
            results_df = pd.DataFrame(results_data)
            st.dataframe(results_df, use_container_width=True)
            
            # Show detailed view if requested
            if show_details:
                st.subheader("📋 Detailed Candidate Information")
                
                for i, candidate in enumerate(candidates[:5], 1):  # Show top 5 detailed
                    similarity = candidate.get('similarity', 0)
                    metadata = candidate.get('metadata', {})
                    document_preview = candidate.get('document', '')[:300]
                    
                    with st.expander(f"🔍 Candidate {i} - {metadata.get('name', 'Unknown')} (Score: {similarity:.3f})"):
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.write(f"**Name:** {metadata.get('name', 'N/A')}")
                            st.write(f"**Title:** {metadata.get('title', 'N/A')}")
                            st.write(f"**Experience:** {metadata.get('experience_years', 0)} years")
                            st.write(f"**Similarity Score:** {similarity:.3f}")
                        
                        with col2:
                            skills = metadata.get('skills', [])
                            if skills:
                                st.write(f"**Skills ({len(skills)}):**")
                                # Show skills in a nice format
                                skills_text = ', '.join(skills[:10])
                                if len(skills) > 10:
                                    skills_text += f" + {len(skills) - 10} more"
                                st.write(skills_text)
                            else:
                                st.write("**Skills:** Not specified")
                        
                        if document_preview:
                            st.write("**Resume Preview:**")
                            st.text(document_preview + "...")
                        
                        # Add action buttons
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            if st.button(f"📧 Contact", key=f"contact_{i}"):
                                st.info("Contact functionality would be implemented here")
                        with col2:
                            if st.button(f"📄 View Full Resume", key=f"resume_{i}"):
                                st.info("Full resume view would be implemented here")
                        with col3:
                            if st.button(f"⭐ Save to Favorites", key=f"favorite_{i}"):
                                st.success("Added to favorites!")
            
        except Exception as e:
            st.error(f"Error displaying results: {str(e)}")
            logger.error(f"Display results error: {str(e)}")
    

    
    def show_all_candidates_summary(self):
        """Show summary of all candidates in the database"""
        try:
            candidates = vector_store.get_all_candidates()
            
            if not candidates:
                st.warning("No candidates in database")
                return
            
            st.subheader(f"📊 All Candidates Summary ({len(candidates)} total)")
            
            # Extract summary data
            summary_data = []
            for candidate in candidates:
                metadata = candidate.get('metadata', {})
                summary_data.append({
                    'Name': metadata.get('name', 'N/A'),
                    'Title': metadata.get('title', 'N/A'),
                    'Experience': f"{metadata.get('experience_years', 0)} years",
                    'Skills': len(metadata.get('skills', [])),
                    'ID': candidate.get('candidate_id', 'Unknown')[:12] + "..."
                })
            
            # Display as dataframe
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_exp = sum(c.get('metadata', {}).get('experience_years', 0) for c in candidates) / len(candidates)
                st.metric("Avg Experience", f"{avg_exp:.1f} years")
            with col2:
                total_skills = sum(len(c.get('metadata', {}).get('skills', [])) for c in candidates)
                st.metric("Total Skills", total_skills)
            with col3:
                avg_skills = total_skills / len(candidates)
                st.metric("Avg Skills/Candidate", f"{avg_skills:.1f}")
            
        except Exception as e:
            st.error(f"Error loading candidate summary: {str(e)}")
            logger.error(f"Candidate summary error: {str(e)}")
    
    def visualize_match_results(self, matches: List[MatchResult]):
        """Create visualizations for match results"""
        # Prepare data
        df = pd.DataFrame([{
            'Candidate': match.candidate_name,
            'Overall Score': match.overall_score,
            'Skills Match': match.skills_match_score,
            'Experience Match': match.experience_match_score,
            'Semantic Similarity': getattr(match, 'semantic_similarity_score', 0.0)
        } for match in matches])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Bar chart of overall scores
            fig_bar = px.bar(
                df, 
                x='Candidate', 
                y='Overall Score',
                title="Overall Match Scores",
                color='Overall Score',
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            # Radar chart for top candidate
            top_candidate = df.iloc[0]
            categories = ['Skills Match', 'Experience Match', 'Semantic Similarity']
            values = [top_candidate[cat] for cat in categories]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=top_candidate['Candidate']
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title=f"Top Candidate: {top_candidate['Candidate']}"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Detailed comparison heatmap
        comparison_df = df.set_index('Candidate')[['Skills Match', 'Experience Match', 'Semantic Similarity']]
        fig_heatmap = px.imshow(
            comparison_df.T,
            title="Candidate Comparison Matrix",
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def display_match_results(self, matches: List[MatchResult]):
        """Display detailed match results"""
        for i, match in enumerate(matches):
            with st.expander(f"#{i+1} {match.candidate_name} - Score: {match.overall_score:.2f}"):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**Matching Skills:**")
                    for skill in match.matching_skills[:10]:  # Show top 10
                        st.write(f"✅ {skill}")
                    
                    st.write("**Strengths:**")
                    for strength in match.strength_areas:
                        st.write(f"💪 {strength}")
                
                with col2:
                    st.write("**Missing Skills:**")
                    for skill in match.missing_skills[:10]:  # Show top 10
                        st.write(f"❌ {skill}")
                    
                    st.write("**Improvement Areas:**")
                    for area in match.improvement_areas:
                        st.write(f"📈 {area}")
                
                st.write("**AI Recommendation:**")
                st.write(match.recommendation)
                
                st.write("**Match Summary:**")
                st.write(match.match_summary)
    
    def display_resume_summary(self, resume_data: ResumeData):
        """Display resume summary with comprehensive safe attribute access"""
        try:
            # Ensure resume_data is valid
            if not resume_data:
                st.error("❌ No resume data available")
                return
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**📋 Contact Information**")
                
                # Initialize profile if None (dataclass default should handle this, but being extra safe)
                if not hasattr(resume_data, 'profile') or resume_data.profile is None:
                    from app.models.resume_data import ProfileInfo
                    resume_data.profile = ProfileInfo()
                
                # Safe attribute access with comprehensive fallbacks
                profile = resume_data.profile
                name = getattr(profile, 'name', '') or 'Unknown Name'
                email = getattr(profile, 'email', '') or 'Not provided'
                phone = getattr(profile, 'phone', '') or 'Not provided'
                location = getattr(profile, 'location', '') or 'Not provided'
                title = getattr(profile, 'title', '') or 'No Title'
                
                # Initialize experience if None
                if not hasattr(resume_data, 'experience') or resume_data.experience is None:
                    from app.models.resume_data import ExperienceInfo
                    resume_data.experience = ExperienceInfo()
                
                # Experience handling
                experience = resume_data.experience
                experience_years = getattr(experience, 'total_years', 0) or 0
                
                st.write(f"**Name:** {name}")
                st.write(f"**Title:** {title}")
                st.write(f"**Email:** {email}")
                st.write(f"**Phone:** {phone}")
                st.write(f"**Location:** {location}")
                st.write(f"**Experience:** {experience_years} years")
            
            with col2:
                st.markdown("**🛠️ Skills & Technologies**")
                
                # Initialize skills if None
                if not hasattr(resume_data, 'skills') or resume_data.skills is None:
                    from app.models.resume_data import SkillsInfo
                    resume_data.skills = SkillsInfo()
                
                # Comprehensive skills handling
                skills_found = False
                skills = resume_data.skills
                
                # Technical skills
                technical_skills = getattr(skills, 'technical', []) or []
                if technical_skills:
                    st.write("**Technical Skills:**")
                    for skill in technical_skills[:5]:  # Limit to top 5
                        st.write(f"• {skill}")
                    skills_found = True
                
                # Soft skills
                soft_skills = getattr(skills, 'soft', []) or []
                if soft_skills:
                    st.write("**Soft Skills:**")
                    for skill in soft_skills[:3]:  # Limit to top 3
                        st.write(f"• {skill}")
                    skills_found = True
                
                # Certifications
                certifications = getattr(skills, 'certifications', []) or []
                if certifications:
                    st.write("**Certifications:**")
                    for cert in certifications[:3]:  # Limit to top 3
                        st.write(f"• {cert}")
                    skills_found = True
                
                if not skills_found:
                    st.write("No skills information available")
            
            # Display summary if available
            summary = getattr(resume_data, 'summary', '') or ''
            if summary and summary.strip():
                st.markdown("**📝 Professional Summary**")
                st.write(summary)
            
            # Additional sections if available
            # Initialize tools_libraries if None
            if not hasattr(resume_data, 'tools_libraries') or resume_data.tools_libraries is None:
                from app.models.resume_data import ToolsLibrariesInfo
                resume_data.tools_libraries = ToolsLibrariesInfo()
            
            tools = resume_data.tools_libraries
            languages = getattr(tools, 'programming_languages', []) or []
            frameworks = getattr(tools, 'frameworks', []) or []
            databases = getattr(tools, 'databases', []) or []
            
            # Show tools & technologies directly (no nested expander)
            if languages or frameworks or databases:
                st.markdown("**🔧 Tools & Technologies**")
                if languages:
                    st.write(f"**Programming Languages:** {', '.join(languages[:5])}")
                if frameworks:
                    st.write(f"**Frameworks:** {', '.join(frameworks[:5])}")
                if databases:
                    st.write(f"**Databases:** {', '.join(databases[:3])}")
            
            # Work experience details - show directly (no nested expander)
            experience = resume_data.experience  # Already initialized above
            companies = getattr(experience, 'companies', []) or []
            roles = getattr(experience, 'roles', []) or []
            achievements = getattr(experience, 'achievements', []) or []
            
            if companies or roles or achievements:
                st.markdown("**💼 Work Experience**")
                if companies:
                    st.write(f"**Companies:** {', '.join(companies[:3])}")
                if roles:
                    st.write(f"**Roles:** {', '.join(roles[:3])}")
                if achievements:
                    st.write("**Key Achievements:**")
                    for achievement in achievements[:3]:
                        st.write(f"• {achievement}")
                
        except Exception as e:
            st.error(f"Error displaying resume summary: {str(e)}")
            logger.error(f"Resume summary display error: {str(e)}")
            
            # Fallback display with minimal information
            try:
                st.markdown("**⚠️ Fallback Display - Basic Information**")
                if hasattr(resume_data, 'profile') and resume_data.profile:
                    name = getattr(resume_data.profile, 'name', 'Unknown')
                    st.write(f"**Name:** {name}")
                st.write("**Status:** Resume processed but some display fields unavailable")
            except:
                st.error("Unable to display any resume information")
    
    def display_job_summary(self, job_data: JobDescription):
        """Display job summary"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"**Company:** {job_data.company}")
            st.write(f"**Required Experience:** {job_data.experience_years} years")
            st.write(f"**Education:** {job_data.education_level}")
        
        with col2:
            st.write("**Required Skills:**")
            for skill in job_data.required_skills[:8]:
                st.write(f"• {skill}")
        
        if job_data.summary:
            st.write("**Summary:**")
            st.write(job_data.summary)
    
    def display_processed_resumes(self):
        """Display list of processed resumes with safe data access and management options"""
        try:
            resumes = asyncio.run(self.resume_processor.list_processed_resumes())
            
            if resumes:
                # Header with management options
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"📋 Processed Resumes ({len(resumes)})")
                with col2:
                    if st.button("🗑️ Clear All Resumes", type="secondary", help="Delete all resume data and vector collections"):
                        if st.session_state.get('confirm_delete_resumes'):
                            self.clear_all_resumes()
                            st.rerun()
                        else:
                            st.session_state.confirm_delete_resumes = True
                            st.warning("⚠️ Click again to confirm deletion of ALL resume data!")
                
                # Create dataframe for display with correct field mapping
                resume_data = []
                for resume in resumes:
                    try:
                        # Map the correct fields from the resume data structure
                        name = resume.get('name', 'Unknown')
                        title = resume.get('title', 'No Title')
                        filename = resume.get('filename', 'Unknown File')
                        
                        # Show name and filename for better identification
                        display_name = name if name != 'Unknown' else filename
                        
                        resume_data.append({
                            'Name': display_name,
                            'Title': title,
                            'Experience': f"{resume.get('experience_years', 0)} years",
                            'Filename': filename,
                            'Processed': resume.get('processed_at', 'Unknown')[:19] if resume.get('processed_at') else 'Unknown'
                        })
                    except Exception as e:
                        logger.error(f"Error processing resume data: {e}")
                        resume_data.append({
                            'Name': 'Error loading',
                            'Title': 'Error',
                            'Experience': '0 years',
                            'Filename': 'Error',
                            'Processed': 'Error'
                        })
                
                if resume_data:
                    df = pd.DataFrame(resume_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Show detailed info for debugging
                    with st.expander("🔍 Debug: Raw Resume Data", expanded=False):
                        st.json(resumes[:2])  # Show first 2 resumes for debugging
                else:
                    st.warning("Resume data could not be loaded properly.")
            else:
                st.info("📝 No resumes processed yet. Upload some resume files to get started!")
        
        except Exception as e:
            st.error(f"Error loading resumes: {str(e)}")
            logger.error(f"Display processed resumes error: {str(e)}")
            
            # Show help information
            with st.expander("🔧 Troubleshooting", expanded=False):
                st.write("**Common issues:**")
                st.write("• API key not set (check OPENAI_API_KEY environment variable)")
                st.write("• ChromaDB initialization issues")
                st.write("• File permission issues in data/ directory")
                st.write("• LangChain service unavailable")
    
    def clear_all_resumes(self):
        """Clear all resume data and vector collections"""
        try:
            import shutil
            from pathlib import Path
            
            # Clear file storage
            resumes_dir = Path("data/resumes")
            if resumes_dir.exists():
                for file in resumes_dir.glob("*.json"):
                    file.unlink()
                st.success("✅ Cleared resume file storage")
            
            # Clear vector database
            try:
                from app.services.vector_store import VectorStore
                vector_store = VectorStore()
                
                # Reset/clear the vector store collections
                vector_store.reset_collections()
                st.success("✅ Cleared vector database collections")
            except Exception as e:
                st.warning(f"Vector database clear failed: {e}")
            
            # Clear cache
            st.cache_data.clear()
            
            # Reset session state
            if 'confirm_delete_resumes' in st.session_state:
                del st.session_state.confirm_delete_resumes
            
            st.success("🎉 All resume data cleared successfully!")
            
        except Exception as e:
            st.error(f"Error clearing resume data: {e}")
            logger.error(f"Clear resumes error: {e}")
    
    def display_stored_jobs(self):
        """Display list of stored jobs - redirects to new clean interface"""
        self.display_jobs_table()
    
    def get_analytics_data(self):
        """Get data for analytics"""
        try:
            resumes = get_processed_resumes()
            jobs = get_stored_jobs()
            
            if not resumes and not jobs:
                return None
            
            return {
                'resumes': resumes,
                'jobs': jobs,
                'total_resumes': len(resumes),
                'total_jobs': len(jobs)
            }
        
        except Exception as e:
            logger.error(f"Error getting analytics data: {str(e)}")
            return None
    
    def display_job_market_overview(self, analytics_data):
        """Display job market overview for candidates"""
        st.subheader("🎯 Job Market Overview")
        
        jobs = analytics_data.get('jobs', [])
        if not jobs:
            st.warning("No job data available. Add some jobs to see market insights.")
            return
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Opportunities", len(jobs))
        
        with col2:
            experience_levels = [job.get('experience_years', 0) for job in jobs]
            avg_exp_required = sum(experience_levels) / len(experience_levels) if experience_levels else 0
            st.metric("Avg Experience Required", f"{avg_exp_required:.1f} years")
        
        with col3:
            companies = set(job.get('company', '').strip() for job in jobs if job.get('company', '').strip())
            st.metric("Unique Companies", len(companies))
        
        with col4:
            locations = set(job.get('location', '').strip() for job in jobs if job.get('location', '').strip())
            st.metric("Job Locations", len(locations))
        
        # Experience level distribution
        col1, col2 = st.columns(2)
        
        with col1:
            if experience_levels:
                # Categorize experience levels
                exp_categories = []
                for exp in experience_levels:
                    if exp <= 2:
                        exp_categories.append("Entry Level (0-2 years)")
                    elif exp <= 5:
                        exp_categories.append("Mid Level (3-5 years)")
                    elif exp <= 10:
                        exp_categories.append("Senior Level (6-10 years)")
                    else:
                        exp_categories.append("Expert Level (10+ years)")
                
                exp_counts = {}
                for cat in exp_categories:
                    exp_counts[cat] = exp_counts.get(cat, 0) + 1
                
                fig_exp = px.pie(
                    values=list(exp_counts.values()),
                    names=list(exp_counts.keys()),
                    title="Job Opportunities by Experience Level"
                )
                st.plotly_chart(fig_exp, use_container_width=True)
        
        with col2:
            # Top hiring companies
            company_counts = {}
            for job in jobs:
                company = job.get('company', '').strip()
                if company:
                    company_counts[company] = company_counts.get(company, 0) + 1
            
            if company_counts:
                top_companies = sorted(company_counts.items(), key=lambda x: x[1], reverse=True)[:8]
                companies_df = pd.DataFrame(top_companies, columns=['Company', 'Open Positions'])
                
                fig_companies = px.bar(
                    companies_df,
                    x='Open Positions',
                    y='Company',
                    orientation='h',
                    title="Top Hiring Companies"
                )
                st.plotly_chart(fig_companies, use_container_width=True)
    
    def display_skills_intelligence(self, analytics_data):
        """Display skills intelligence and market demand"""
        st.subheader("🛠️ Skills Intelligence & Market Demand")
        
        jobs = analytics_data.get('jobs', [])
        if not jobs:
            st.warning("No job data available for skills analysis.")
            return
        
        # Analyze required skills across all jobs
        all_required_skills = {}
        all_preferred_skills = {}
        
        for job in jobs:
            # Required skills
            required_skills = job.get('required_skills', [])
            for skill in required_skills:
                if skill and skill.strip():
                    skill_name = skill.strip()
                    all_required_skills[skill_name] = all_required_skills.get(skill_name, 0) + 1
            
            # Preferred skills
            preferred_skills = job.get('preferred_skills', [])
            for skill in preferred_skills:
                if skill and skill.strip():
                    skill_name = skill.strip()
                    all_preferred_skills[skill_name] = all_preferred_skills.get(skill_name, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Most In-Demand Skills")
            if all_required_skills:
                top_required = sorted(all_required_skills.items(), key=lambda x: x[1], reverse=True)[:12]
                required_df = pd.DataFrame(top_required, columns=['Skill', 'Job Postings'])
                
                fig_required = px.bar(
                    required_df,
                    x='Job Postings',
                    y='Skill',
                    orientation='h',
                    title="Required Skills Across All Jobs",
                    color='Job Postings',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig_required, use_container_width=True)
            else:
                st.info("No required skills data available")
        
        with col2:
            st.markdown("### ⭐ Preferred Skills (Competitive Advantage)")
            if all_preferred_skills:
                top_preferred = sorted(all_preferred_skills.items(), key=lambda x: x[1], reverse=True)[:12]
                preferred_df = pd.DataFrame(top_preferred, columns=['Skill', 'Job Postings'])
                
                fig_preferred = px.bar(
                    preferred_df,
                    x='Job Postings',
                    y='Skill',
                    orientation='h',
                    title="Preferred Skills for Competitive Edge",
                    color='Job Postings',
                    color_continuous_scale='plasma'
                )
                st.plotly_chart(fig_preferred, use_container_width=True)
            else:
                st.info("No preferred skills data available")
        
        # Skills gap analysis
        resumes = analytics_data.get('resumes', [])
        if resumes and all_required_skills:
            st.markdown("### 📊 Skills Gap Analysis")
            
            # Get candidate skills
            candidate_skills = {}
            for resume in resumes:
                if isinstance(resume, dict):
                    resume_skills = resume.get('skills', []) or []
                else:
                    resume_skills = getattr(resume, 'skills', []) or []
                
                for skill in resume_skills:
                    if skill and skill.strip():
                        skill_name = skill.strip()
                        candidate_skills[skill_name] = candidate_skills.get(skill_name, 0) + 1
            
            # Compare market demand vs candidate supply
            gap_analysis = []
            for skill, market_demand in list(all_required_skills.items())[:15]:
                candidate_supply = candidate_skills.get(skill, 0)
                gap_ratio = market_demand / max(candidate_supply, 1)
                
                gap_analysis.append({
                    'Skill': skill,
                    'Market Demand': market_demand,
                    'Candidate Supply': candidate_supply,
                    'Opportunity Score': gap_ratio
                })
            
            gap_df = pd.DataFrame(gap_analysis)
            gap_df = gap_df.sort_values('Opportunity Score', ascending=False)
            
            st.markdown("**💡 Skills with High Opportunity (Low Supply, High Demand):**")
            st.dataframe(gap_df, use_container_width=True)
    
    def display_career_insights(self, analytics_data):
        """Display career development insights"""
        st.subheader("📈 Career Development Insights")
        
        jobs = analytics_data.get('jobs', [])
        if not jobs:
            st.warning("No job data available for career insights.")
            return
        
        # Career progression analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Entry Points by Experience Level")
            
            # Categorize jobs by experience level and show typical roles
            experience_roles = {
                "Entry Level (0-2 years)": [],
                "Mid Level (3-5 years)": [],
                "Senior Level (6-10 years)": [],
                "Expert Level (10+ years)": []
            }
            
            for job in jobs:
                exp = job.get('experience_years', 0)
                title = job.get('title', 'Unknown Role')
                
                if exp <= 2:
                    experience_roles["Entry Level (0-2 years)"].append(title)
                elif exp <= 5:
                    experience_roles["Mid Level (3-5 years)"].append(title)
                elif exp <= 10:
                    experience_roles["Senior Level (6-10 years)"].append(title)
                else:
                    experience_roles["Expert Level (10+ years)"].append(title)
            
            for level, roles in experience_roles.items():
                if roles:
                    st.markdown(f"**{level}**")
                    role_counts = {}
                    for role in roles:
                        role_counts[role] = role_counts.get(role, 0) + 1
                    
                    top_roles = sorted(role_counts.items(), key=lambda x: x[1], reverse=True)[:3]
                    for role, count in top_roles:
                        st.write(f"• {role} ({count} openings)")
                    st.divider()
        
        with col2:
            st.markdown("### 🏢 Work Arrangements")
            
            # Analyze location preferences
            location_analysis = {}
            remote_count = 0
            
            for job in jobs:
                location = job.get('location', '').strip().lower()
                if not location or location in ['remote', 'work from home', 'wfh']:
                    remote_count += 1
                elif location:
                    location_analysis[location.title()] = location_analysis.get(location.title(), 0) + 1
            
            if remote_count > 0:
                st.metric("Remote Opportunities", remote_count)
            
            if location_analysis:
                top_locations = sorted(location_analysis.items(), key=lambda x: x[1], reverse=True)[:5]
                st.markdown("**Top Office Locations:**")
                for location, count in top_locations:
                    st.write(f"• {location}: {count} jobs")
        
        # Education requirements analysis
        st.markdown("### 🎓 Education Requirements Analysis")
        
        education_requirements = {}
        for job in jobs:
            edu = job.get('education_level', '').strip()
            if edu:
                education_requirements[edu] = education_requirements.get(edu, 0) + 1
        
        if education_requirements:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                edu_df = pd.DataFrame(list(education_requirements.items()), columns=['Education Level', 'Job Count'])
                fig_edu = px.bar(
                    edu_df,
                    x='Job Count',
                    y='Education Level',
                    orientation='h',
                    title="Education Requirements in Job Market"
                )
                st.plotly_chart(fig_edu, use_container_width=True)
            
            with col2:
                st.markdown("**Education Insights:**")
                total_jobs = len(jobs)
                for edu, count in sorted(education_requirements.items(), key=lambda x: x[1], reverse=True)[:5]:
                    percentage = (count / total_jobs) * 100
                    st.write(f"• {edu}: {percentage:.1f}%")
    
    def display_company_analysis(self, analytics_data):
        """Display company and industry analysis"""
        st.subheader("🏢 Company & Industry Analysis")
        
        jobs = analytics_data.get('jobs', [])
        if not jobs:
            st.warning("No job data available for company analysis.")
            return
        
        # Company analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏢 Top Hiring Companies")
            
            company_analysis = {}
            for job in jobs:
                company = job.get('company', '').strip()
                if company:
                    if company not in company_analysis:
                        company_analysis[company] = {
                            'job_count': 0,
                            'avg_experience': 0,
                            'total_experience': 0,
                            'roles': []
                        }
                    
                    company_analysis[company]['job_count'] += 1
                    company_analysis[company]['total_experience'] += job.get('experience_years', 0)
                    company_analysis[company]['roles'].append(job.get('title', 'Unknown'))
            
            # Calculate averages
            for company_data in company_analysis.values():
                if company_data['job_count'] > 0:
                    company_data['avg_experience'] = company_data['total_experience'] / company_data['job_count']
            
            # Display top companies
            top_companies = sorted(company_analysis.items(), key=lambda x: x[1]['job_count'], reverse=True)[:8]
            
            company_table = []
            for company, data in top_companies:
                company_table.append({
                    'Company': company,
                    'Open Positions': data['job_count'],
                    'Avg Experience Required': f"{data['avg_experience']:.1f} years",
                    'Variety of Roles': len(set(data['roles']))
                })
            
            companies_df = pd.DataFrame(company_table)
            st.dataframe(companies_df, use_container_width=True)
        
        with col2:
            st.markdown("### 📊 Market Insights")
            
            # Company size analysis (based on number of openings)
            size_categories = {
                'Small (1-2 openings)': 0,
                'Medium (3-5 openings)': 0,
                'Large (6+ openings)': 0
            }
            
            for company, data in company_analysis.items():
                job_count = data['job_count']
                if job_count <= 2:
                    size_categories['Small (1-2 openings)'] += 1
                elif job_count <= 5:
                    size_categories['Medium (3-5 openings)'] += 1
                else:
                    size_categories['Large (6+ openings)'] += 1
            
            fig_size = px.pie(
                values=list(size_categories.values()),
                names=list(size_categories.keys()),
                title="Companies by Hiring Volume"
            )
            st.plotly_chart(fig_size, use_container_width=True)
        
        # Industry skills analysis
        st.markdown("### 🔬 Industry Skills Trends")
        
        # Group similar companies and analyze their skill requirements
        all_job_skills = []
        for job in jobs:
            company = job.get('company', '').strip()
            required_skills = job.get('required_skills', [])
            preferred_skills = job.get('preferred_skills', [])
            
            for skill in required_skills + preferred_skills:
                if skill and skill.strip():
                    all_job_skills.append({
                        'Company': company,
                        'Skill': skill.strip(),
                        'Job Title': job.get('title', 'Unknown')
                    })
        
        if all_job_skills:
            skills_df = pd.DataFrame(all_job_skills)
            
            # Find most versatile skills (appear across many companies)
            skill_company_count = skills_df.groupby('Skill')['Company'].nunique().reset_index()
            skill_company_count.columns = ['Skill', 'Companies Using Skill']
            skill_company_count = skill_company_count.sort_values('Companies Using Skill', ascending=False).head(10)
            
            st.markdown("**🌟 Most Versatile Skills (High Cross-Company Demand):**")
            st.dataframe(skill_company_count, use_container_width=True)
    
    def get_available_skills(self):
        """Get list of all available skills"""
        try:
            resumes = asyncio.run(self.resume_processor.list_processed_resumes())
            all_skills = set()
            
            for resume in resumes:
                # Handle both dictionary and object formats
                if isinstance(resume, dict):
                    resume_skills = resume.get('skills', []) or []
                else:
                    resume_skills = getattr(resume, 'skills', []) or []
                
                if isinstance(resume_skills, list):
                    for skill in resume_skills:
                        if skill:
                            all_skills.add(str(skill).strip())
            
            return sorted(list(all_skills))
        
        except Exception as e:
            logger.error(f"Error getting available skills: {str(e)}")
            return []
    



def main():
    """Main entry point"""
    try:
        app = StreamlitApp()
        app.run()
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Streamlit app error: {str(e)}")


if __name__ == "__main__":
    main()
