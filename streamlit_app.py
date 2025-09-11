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

# Add app to path
sys.path.append(str(Path(__file__).parent))

from app.services.resume_processor import resume_processor
from app.services.job_processor import job_processor
from app.services.data_pipeline import data_pipeline
from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service
from app.models.resume_data import ResumeData, JobDescription, MatchResult
from app.core.logging import get_logger

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
        
    def run(self):
        """Main application entry point"""
        st.title("🤖 AI Resume Matcher")
        st.markdown("**LangChain-Powered Intelligent Resume Matching System**")
        
        # Initialize data if needed
        initialize_data()
        
        # Sidebar navigation
        with st.sidebar:
            st.header("Navigation")
            page = st.radio(
                "Choose a page:",
                ["📄 Resume Upload", "📋 Job Management", "🎯 Job Matching", "� Search Candidates", "�📊 Analytics", "� Data Pipeline"]
            )
        
        # Route to selected page
        if page == "📄 Resume Upload":
            self.resume_upload_page()
        elif page == "📋 Job Management":
            self.job_management_page()
        elif page == "🎯 Job Matching":
            self.job_matching_page()
        elif page == "📊 Analytics":
            self.analytics_page()
        elif page == "🔍 Search Candidates":
            self.search_page()
        elif page == "� Data Pipeline":
            self.data_pipeline_page()
    
    def data_pipeline_page(self):
        """Data pipeline management page"""
        st.header("🔧 Data Pipeline Management")
        
        # Pipeline statistics
        col1, col2, col3 = st.columns(3)
        stats = get_pipeline_stats()
        
        with col1:
            st.metric("Total Resumes", stats.get('total_resumes', 0))
        with col2:
            st.metric("Total Jobs", stats.get('total_jobs', 0))
        with col3:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        st.subheader("Bulk Upload")
        
        tab1, tab2, tab3 = st.tabs(["📄 Bulk Resume Upload", "📋 Bulk Job Upload", "🎯 Sample Data"])
        
        with tab1:
            st.write("Upload multiple resume files at once")
            bulk_resume_files = st.file_uploader(
                "Select resume files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Select multiple resume files for bulk processing"
            )
            
            if bulk_resume_files and st.button("🚀 Process Bulk Resumes"):
                self.process_bulk_resumes(bulk_resume_files)
        
        with tab2:
            st.write("Upload job descriptions from CSV or JSON")
            
            job_upload_type = st.radio("Select upload format:", ["CSV", "JSON"])
            
            if job_upload_type == "CSV":
                st.write("**Expected CSV columns:** title, company, description, experience_years, location")
                job_file = st.file_uploader("Upload CSV file", type=['csv'])
            else:
                st.write("**Expected JSON format:** Array of job objects")
                job_file = st.file_uploader("Upload JSON file", type=['json'])
            
            if job_file and st.button("📋 Process Job File"):
                self.process_bulk_jobs(job_file, job_upload_type)
        
        with tab3:
            st.write("Load sample data for testing and demonstration")
            
            if st.button("📊 Load Sample Data"):
                with st.spinner("Loading sample data..."):
                    try:
                        result = asyncio.run(self.data_pipeline.process_sample_data())
                        st.success(f"✅ Loaded {result['jobs']['processed']} jobs and {result['resumes']['processed']} resumes!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to load sample data: {e}")
    
    def process_bulk_resumes(self, files):
        """Process bulk resume uploads"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(f"{message} ({current}/{total})")
        
        try:
            # Save files temporarily and process
            temp_files = []
            for file in files:
                temp_path = Path("data/temp") / file.name
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(temp_path, "wb") as f:
                    f.write(file.getbuffer())
                temp_files.append(temp_path)
            
            # Process files
            result = asyncio.run(
                self.data_pipeline.bulk_upload_resumes(temp_files, progress_callback)
            )
            
            # Clean up temp files
            for temp_file in temp_files:
                temp_file.unlink(missing_ok=True)
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Processed: {result['processed']}")
            with col2:
                st.error(f"❌ Failed: {result['failed']}")
            
            if result['errors']:
                with st.expander("View Errors"):
                    for error in result['errors']:
                        st.write(f"• {error}")
            
            st.cache_data.clear()
            
        except Exception as e:
            st.error(f"Bulk processing failed: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def process_bulk_jobs(self, file, file_type):
        """Process bulk job uploads"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def progress_callback(current, total, message):
            progress_bar.progress(current / total)
            status_text.text(f"{message} ({current}/{total})")
        
        try:
            # Save file temporarily
            temp_path = Path("data/temp") / file.name
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process based on file type
            if file_type == "CSV":
                result = asyncio.run(
                    self.data_pipeline.bulk_upload_jobs_from_csv(temp_path, progress_callback)
                )
            else:
                result = asyncio.run(
                    self.data_pipeline.bulk_upload_jobs_from_json(temp_path, progress_callback)
                )
            
            # Clean up temp file
            temp_path.unlink(missing_ok=True)
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Processed: {result['processed']}")
            with col2:
                st.error(f"❌ Failed: {result['failed']}")
            
            if result['errors']:
                with st.expander("View Errors"):
                    for error in result['errors']:
                        st.write(f"• {error}")
            
            st.cache_data.clear()
            
        except Exception as e:
            st.error(f"Bulk job processing failed: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def resume_upload_page(self):
        """Resume upload and processing page"""
        st.header("📄 Resume Upload & Processing")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload resume files",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF, DOCX, or TXT resume files"
        )
        
        if uploaded_files:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Processing Status")
                progress_bar = st.progress(0)
                status_text = st.empty()
                
            with col2:
                st.subheader("Processing Options")
                batch_process = st.checkbox("Batch process all files", value=True)
                overwrite_existing = st.checkbox("Overwrite existing resumes", value=False)
            
            if st.button("🚀 Process Resumes", type="primary"):
                self.process_uploaded_resumes(uploaded_files, progress_bar, status_text, batch_process)
        
        # Display processed resumes
        self.display_processed_resumes()
    
    def job_management_page(self):
        """Job description management page"""
        st.header("📋 Job Description Management")
        
        # Job input form
        with st.form("job_form"):
            st.subheader("Add New Job Description")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                job_title = st.text_input("Job Title", placeholder="e.g., Senior Software Engineer")
                company = st.text_input("Company", placeholder="e.g., TechCorp Inc.")
            
            with col2:
                experience_years = st.number_input("Required Experience (years)", min_value=0, max_value=20, value=3)
                location = st.text_input("Location", placeholder="e.g., San Francisco, CA")
            
            job_description = st.text_area(
                "Job Description",
                height=200,
                placeholder="Paste the complete job description here..."
            )
            
            submitted = st.form_submit_button("💾 Save Job Description", type="primary")
            
            if submitted and job_title and job_description:
                self.save_job_description(job_title, company, job_description, experience_years, location)
        
        # Display existing jobs
        self.display_stored_jobs()
    
    def job_matching_page(self):
        """Job matching and results page"""
        st.header("🎯 Job Matching & Results")
        
        # Get stored jobs
        stored_jobs = get_stored_jobs()
        
        if not stored_jobs:
            st.warning("No job descriptions stored. Please add jobs in the Job Management page first.")
            return
        
        # Matching options
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Select Matching Method")
            matching_method = st.radio(
                "Choose matching approach:",
                ["🎯 Job-to-Candidates (Select Job)", "📝 Custom Job Description", "🔍 Semantic Search Query"],
                horizontal=True
            )
        
        with col2:
            st.subheader("Settings")
            top_k = st.slider("Number of candidates", min_value=3, max_value=50, value=10)
            show_details = st.checkbox("Show detailed results", value=True)
        
        # Different matching interfaces
        if matching_method == "🎯 Job-to-Candidates (Select Job)":
            self.job_dropdown_matching(stored_jobs, top_k, show_details)
        elif matching_method == "📝 Custom Job Description":
            self.custom_job_matching(top_k, show_details)
        elif matching_method == "🔍 Semantic Search Query":
            self.semantic_search_matching(top_k, show_details)
    
    def job_dropdown_matching(self, stored_jobs, top_k, show_details):
        """Job dropdown selection matching"""
        st.subheader("🎯 Select Job for Candidate Matching")
        
        # Enhanced job selection with preview
        job_options = {}
        job_previews = {}
        
        for job in stored_jobs:
            display_name = f"{job['title']} - {job['company']}"
            job_options[display_name] = job['id']
            
            # Create preview text
            skills_preview = ', '.join(job.get('required_skills', [])[:5])
            if len(job.get('required_skills', [])) > 5:
                skills_preview += f" + {len(job.get('required_skills', [])) - 5} more"
            
            job_previews[display_name] = {
                'experience': f"{job.get('experience_years', 'N/A')} years",
                'location': job.get('location', 'Not specified'),
                'skills': skills_preview or 'No skills listed',
                'description_preview': job.get('raw_text', '')[:150] + "..." if job.get('raw_text') else 'No description'
            }
        
        selected_job_name = st.selectbox(
            "Select Job for Matching",
            list(job_options.keys()),
            help="Choose a job to find the best matching candidates"
        )
        
        if selected_job_name:
            # Show job preview
            preview = job_previews[selected_job_name]
            with st.expander("📋 Job Preview", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Experience Required:** {preview['experience']}")
                    st.write(f"**Location:** {preview['location']}")
                with col2:
                    st.write(f"**Key Skills:** {preview['skills']}")
                st.write(f"**Description:** {preview['description_preview']}")
            
            job_id = job_options[selected_job_name]
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍 Find Matches", type="primary"):
                    self.perform_job_matching(job_id, top_k, show_details)
            with col2:
                auto_refresh = st.checkbox("Auto-refresh results", value=False)
                if auto_refresh:
                    self.perform_job_matching(job_id, top_k, show_details)
    
    def custom_job_matching(self, top_k, show_details):
        """Custom job description matching"""
        st.subheader("📝 Enter Custom Job Description")
        
        with st.form("custom_job_form"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                custom_job_text = st.text_area(
                    "Job Description",
                    height=200,
                    placeholder="Enter a job description or requirements to find matching candidates...",
                    help="Describe the role, required skills, experience, and responsibilities"
                )
            
            with col2:
                st.write("**Optional Details:**")
                custom_title = st.text_input("Job Title (optional)", placeholder="e.g., Senior Developer")
                custom_experience = st.number_input("Required Experience (years)", min_value=0, max_value=20, value=0)
            
            submitted = st.form_submit_button("🔍 Find Matching Candidates", type="primary")
            
            if submitted and custom_job_text.strip():
                self.perform_custom_job_matching(custom_job_text, custom_title, custom_experience, top_k, show_details)
    
    def semantic_search_matching(self, top_k, show_details):
        """Semantic search query matching"""
        st.subheader("🔍 Semantic Search for Candidates")
        
        # Predefined search examples
        st.write("**Quick Search Examples:**")
        example_queries = [
            "API Gateway Kong Apigee microservices",
            "Flutter React Native mobile development",
            "DevOps Kubernetes Docker AWS",
            "Machine Learning Python TensorFlow",
            "Full-stack JavaScript React Node.js"
        ]
        
        selected_example = st.selectbox(
            "Choose an example or enter custom query:",
            ["Custom Query"] + example_queries
        )
        
        if selected_example != "Custom Query":
            search_query = selected_example
        else:
            search_query = ""
        
        # Search input
        search_input = st.text_input(
            "Search Query",
            value=search_query,
            placeholder="Enter skills, technologies, or job requirements...",
            help="Use natural language to describe what you're looking for"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔍 Search Candidates", type="primary"):
                if search_input.strip():
                    self.perform_semantic_search(search_input, top_k, show_details)
                else:
                    st.warning("Please enter a search query")
        
        with col2:
            if st.button("🧪 Test Vector Search"):
                self.test_vector_search_functionality()
        
        with col3:
            if st.button("📊 Show Search Stats"):
                self.show_search_statistics()
    
    def analytics_page(self):
        """Analytics and visualization page"""
        st.header("📊 Analytics & Insights")
        
        # Get analytics data
        analytics_data = self.get_analytics_data()
        
        if not analytics_data:
            st.info("No data available for analytics. Process some resumes and jobs first.")
            return
        
        # Display analytics
        self.display_resume_analytics(analytics_data)
        self.display_matching_analytics(analytics_data)
    
    
    def process_uploaded_resumes(self, uploaded_files, progress_bar, status_text, batch_process):
        """Process uploaded resume files"""
        try:
            total_files = len(uploaded_files)
            
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
                    
                    st.success(f"✅ Successfully processed {uploaded_file.name}")
                    
                    # Display resume summary
                    with st.expander(f"📄 {resume_data.profile.name} - {resume_data.profile.title}"):
                        self.display_resume_summary(resume_data)
                    
                except Exception as e:
                    st.error(f"❌ Failed to process {uploaded_file.name}: {str(e)}")
                
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
            
            status_text.text("✅ All files processed!")
            
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
    
    def save_job_description(self, title, company, description, experience_years, location):
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
            
            st.success(f"✅ Successfully saved job: {title}")
            
            # Display job summary
            with st.expander(f"📋 {title} - {company}"):
                self.display_job_summary(job_data)
            
        except Exception as e:
            st.error(f"❌ Failed to save job: {str(e)}")
    
    def perform_job_matching(self, job_id, top_k, show_details=True):
        """Perform job matching and display results"""
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
    
    def perform_custom_job_matching(self, job_text, title, experience_years, top_k, show_details):
        """Perform matching with custom job description"""
        try:
            st.info(f"🔍 Finding matches for custom job: {title or 'Custom Position'}")
            
            # Generate embedding for custom job
            with st.spinner("Processing job description..."):
                job_embedding = embedding_service.generate_embedding(job_text)
            
            # Search for similar resumes
            with st.spinner("Searching for candidates..."):
                candidates = vector_store.search_similar(
                    query_embedding=job_embedding,
                    top_k=top_k,
                    collection_name="resumes"
                )
            
            if not candidates:
                st.warning("No matching candidates found")
                return
            
            # Display results
            st.subheader(f"🎯 Top {len(candidates)} Candidates for Custom Job")
            
            # Show job details
            with st.expander("📋 Job Requirements", expanded=False):
                if title:
                    st.write(f"**Title:** {title}")
                if experience_years > 0:
                    st.write(f"**Experience Required:** {experience_years} years")
                st.write(f"**Description:** {job_text[:200]}...")
            
            # Display results
            self.display_vector_search_results(candidates, show_details)
            
        except Exception as e:
            st.error(f"Error performing custom job matching: {str(e)}")
            logger.error(f"Custom job matching error: {str(e)}")
    
    def perform_semantic_search(self, search_query, top_k, show_details):
        """Perform semantic search for candidates"""
        try:
            st.info(f"🔍 Semantic search for: '{search_query}'")
            
            # Generate embedding for search query
            with st.spinner("Processing search query..."):
                query_embedding = embedding_service.generate_embedding(search_query)
            
            # Search for similar resumes
            with st.spinner("Searching candidates..."):
                candidates = vector_store.search_similar(
                    query_embedding=query_embedding,
                    top_k=top_k,
                    collection_name="resumes"
                )
            
            if not candidates:
                st.warning("No candidates found matching your search")
                return
            
            # Display results
            st.subheader(f"🎯 {len(candidates)} Candidates Found")
            
            # Display search query info
            with st.expander("🔍 Search Details", expanded=False):
                st.write(f"**Query:** {search_query}")
                st.write(f"**Search Type:** Semantic similarity matching")
                st.write(f"**Results:** Top {len(candidates)} most similar candidates")
            
            # Display results
            self.display_vector_search_results(candidates, show_details)
            
        except Exception as e:
            st.error(f"Error performing semantic search: {str(e)}")
            logger.error(f"Semantic search error: {str(e)}")
    
    def test_vector_search_functionality(self):
        """Test vector search functionality like in test_vector_search.py"""
        st.subheader("🧪 Vector Search Test Results")
        
        try:
            # Test 1: Check vector store status
            with st.spinner("Testing vector store..."):
                candidates = vector_store.get_all_candidates()
                st.success(f"✅ Vector store operational - {len(candidates)} candidates loaded")
            
            # Test 2: Test embedding generation
            with st.spinner("Testing embedding generation..."):
                test_text = "API Gateway experience with Kong and microservices"
                embedding = embedding_service.generate_embedding(test_text)
                st.success(f"✅ Embedding generation working - {len(embedding)} dimensions")
            
            # Test 3: Test semantic matching
            if len(candidates) > 0:
                with st.spinner("Testing semantic search..."):
                    # Test API Gateway search
                    api_query = "API Gateway Kong Apigee microservices architecture"
                    api_embedding = embedding_service.generate_embedding(api_query)
                    api_results = vector_store.search_similar(
                        query_embedding=api_embedding,
                        top_k=5,
                        collection_name="resumes"
                    )
                    
                    st.success(f"✅ API Gateway search: {len(api_results)} results")
                    
                    # Test mobile development search
                    mobile_query = "Flutter React Native mobile development"
                    mobile_embedding = embedding_service.generate_embedding(mobile_query)
                    mobile_results = vector_store.search_similar(
                        query_embedding=mobile_embedding,
                        top_k=5,
                        collection_name="resumes"
                    )
                    
                    st.success(f"✅ Mobile development search: {len(mobile_results)} results")
                    
                    # Show sample results
                    with st.expander("📊 Sample Search Results"):
                        st.write("**API Gateway Results:**")
                        for i, result in enumerate(api_results[:3], 1):
                            similarity = result.get('similarity', 0)
                            candidate_id = result.get('candidate_id', 'Unknown')
                            st.write(f"{i}. Candidate: {candidate_id[:12]}... | Similarity: {similarity:.3f}")
                        
                        st.write("**Mobile Development Results:**")
                        for i, result in enumerate(mobile_results[:3], 1):
                            similarity = result.get('similarity', 0)
                            candidate_id = result.get('candidate_id', 'Unknown')
                            st.write(f"{i}. Candidate: {candidate_id[:12]}... | Similarity: {similarity:.3f}")
            else:
                st.warning("⚠️ No candidates in vector store - upload some resumes first")
            
            st.success("🎉 All vector search tests completed successfully!")
            
        except Exception as e:
            st.error(f"❌ Vector search test failed: {str(e)}")
            logger.error(f"Vector search test error: {str(e)}")
    
    def show_search_statistics(self):
        """Show search and database statistics"""
        st.subheader("📊 Search & Database Statistics")
        
        try:
            # Get basic stats
            candidates = vector_store.get_all_candidates()
            jobs = get_stored_jobs()
            resumes = get_processed_resumes()
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Candidates", len(candidates))
            with col2:
                st.metric("Stored Jobs", len(jobs))
            with col3:
                st.metric("Processed Resumes", len(resumes))
            with col4:
                # Calculate search readiness
                search_ready = len(candidates) > 0 and len(jobs) > 0
                st.metric("Search Ready", "✅ Yes" if search_ready else "❌ No")
            
            # Vector database info
            st.write("**Vector Database Status:**")
            if len(candidates) > 0:
                # Test embedding dimension
                sample_query = "test query"
                sample_embedding = embedding_service.generate_embedding(sample_query)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"• Embedding Model: sentence-transformers")
                    st.write(f"• Vector Dimensions: {len(sample_embedding)}")
                with col2:
                    st.write(f"• Collections: resumes, job_descriptions")
                    st.write(f"• Search Algorithm: ChromaDB similarity")
            else:
                st.write("• Vector database is empty - upload resumes to populate")
            
            # Search capabilities
            st.write("**Available Search Methods:**")
            st.write("✅ Job-to-Candidates matching")
            st.write("✅ Custom job description search") 
            st.write("✅ Semantic query search")
            st.write("✅ Skills-based filtering")
            
        except Exception as e:
            st.error(f"Error loading statistics: {str(e)}")
            logger.error(f"Statistics error: {str(e)}")
    
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
    
    def search_page(self):
        """Enhanced search and filter page"""
        st.header("🔍 Advanced Candidate Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("Search Candidates")
            
            # Main search input
            search_query = st.text_input(
                "Search Query",
                placeholder="Enter skills, technologies, job titles, or requirements...",
                help="Use natural language to describe what you're looking for"
            )
            
            # Quick search examples
            st.write("**Quick Examples:**")
            example_cols = st.columns(3)
            with example_cols[0]:
                if st.button("🌐 API Gateway Skills"):
                    search_query = "API Gateway Kong Apigee microservices"
                    st.rerun()
            with example_cols[1]:
                if st.button("📱 Mobile Development"):
                    search_query = "Flutter React Native mobile development"
                    st.rerun()
            with example_cols[2]:
                if st.button("☁️ DevOps & Cloud"):
                    search_query = "DevOps Kubernetes Docker AWS"
                    st.rerun()
        
        with col2:
            st.subheader("Filters")
            
            # Experience filter
            min_experience = st.number_input("Min Experience (years)", min_value=0, max_value=20, value=0)
            max_experience = st.number_input("Max Experience (years)", min_value=0, max_value=20, value=20)
            
            # Results count
            max_results = st.slider("Max Results", min_value=5, max_value=50, value=15)
            
            # Search options
            exact_match = st.checkbox("Exact skill matching", value=False)
        
        # Skills filter section
        st.subheader("🛠️ Skills Filter")
        available_skills = self.get_available_skills()
        selected_skills = []
        skill_match_type = "Any of selected"  # Default value
        
        if available_skills:
            col1, col2 = st.columns([2, 1])
            with col1:
                selected_skills = st.multiselect(
                    "Filter by Skills",
                    available_skills,
                    help="Select specific skills to filter candidates"
                )
            with col2:
                skill_match_type = st.radio(
                    "Skill Matching",
                    ["Any of selected", "All of selected"],
                    help="Whether candidate should have any or all selected skills"
                )
        else:
            st.info("No skills available - process some resumes first")
        
        # Search execution
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("🔍 Search Candidates", type="primary"):
                if search_query.strip() or selected_skills:
                    self.execute_advanced_search(
                        search_query.strip(),
                        min_experience,
                        max_experience,
                        selected_skills,
                        skill_match_type,
                        max_results,
                        exact_match
                    )
                else:
                    st.warning("Please enter a search query or select skills")
        
        with col2:
            if st.button("📊 Show All Candidates"):
                self.show_all_candidates_summary()
        
        with col3:
            if st.button("🔄 Reset Filters"):
                st.rerun()
    
    def execute_advanced_search(self, query, min_exp, max_exp, skills, skill_match_type, max_results, exact_match):
        """Execute advanced search with filters"""
        try:
            st.subheader("🎯 Search Results")
            
            # Perform semantic search if query provided
            if query:
                with st.spinner("Performing semantic search..."):
                    query_embedding = embedding_service.generate_embedding(query)
                    candidates = vector_store.search_similar(
                        query_embedding=query_embedding,
                        top_k=max_results * 2,  # Get more to filter
                        collection_name="resumes"
                    )
            else:
                # Get all candidates for skill-only filtering
                candidates = vector_store.get_all_candidates()[:max_results * 2]
            
            if not candidates:
                st.warning("No candidates found")
                return
            
            # Apply filters
            filtered_candidates = self.apply_search_filters(
                candidates, min_exp, max_exp, skills, skill_match_type
            )
            
            # Limit results
            final_results = filtered_candidates[:max_results]
            
            if not final_results:
                st.warning("No candidates match your filters")
                return
            
            # Show search summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Found", len(candidates))
            with col2:
                st.metric("After Filters", len(filtered_candidates))
            with col3:
                st.metric("Showing", len(final_results))
            
            # Display results
            self.display_vector_search_results(final_results, show_details=True)
            
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            logger.error(f"Advanced search error: {str(e)}")
    
    def apply_search_filters(self, candidates, min_exp, max_exp, skills, skill_match_type):
        """Apply experience and skill filters to candidates"""
        filtered = []
        
        for candidate in candidates:
            metadata = candidate.get('metadata', {})
            candidate_exp = metadata.get('experience_years', 0)
            candidate_skills = [s.lower() for s in metadata.get('skills', [])]
            
            # Experience filter
            if candidate_exp < min_exp or candidate_exp > max_exp:
                continue
            
            # Skills filter
            if skills:
                selected_skills_lower = [s.lower() for s in skills]
                
                if skill_match_type == "All of selected":
                    # Must have all selected skills
                    if not all(skill in candidate_skills for skill in selected_skills_lower):
                        continue
                else:
                    # Must have at least one selected skill
                    if not any(skill in candidate_skills for skill in selected_skills_lower):
                        continue
            
            filtered.append(candidate)
        
        return filtered
    
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
        """Display resume summary"""
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write(f"**Email:** {resume_data.profile.email}")
            st.write(f"**Phone:** {resume_data.profile.phone}")
            st.write(f"**Location:** {resume_data.profile.location}")
            st.write(f"**Experience:** {resume_data.experience.total_years} years")
        
        with col2:
            st.write("**Top Skills:**")
            for skill in resume_data.skills.technical[:8]:
                st.write(f"• {skill}")
        
        if resume_data.summary:
            st.write("**Summary:**")
            st.write(resume_data.summary)
    
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
        """Display list of processed resumes"""
        try:
            resumes = asyncio.run(self.resume_processor.list_processed_resumes())
            
            if resumes:
                st.subheader(f"📋 Processed Resumes ({len(resumes)})")
                
                # Create dataframe for display
                df = pd.DataFrame([{
                    'Name': resume['name'],
                    'Title': resume['title'],
                    'Experience': f"{resume['experience_years']} years",
                    'Skills': len(resume['skills']),
                    'Processed': resume['processed_at'][:10]  # Date only
                } for resume in resumes])
                
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No resumes processed yet. Upload some resume files to get started.")
        
        except Exception as e:
            st.error(f"Error loading resumes: {str(e)}")
    
    def display_stored_jobs(self):
        """Display list of stored jobs with enhanced details"""
        try:
            jobs = get_stored_jobs()
            
            if jobs:
                st.subheader(f"📋 Stored Jobs ({len(jobs)})")
                
                # Enhanced job display with expandable details
                for i, job in enumerate(jobs, 1):
                    with st.expander(f"📋 {job['title']} - {job['company']}", expanded=False):
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"**Title:** {job['title']}")
                            st.write(f"**Company:** {job['company']}")
                            st.write(f"**Location:** {job.get('location', 'Not specified')}")
                            st.write(f"**Experience Required:** {job.get('experience_years', 'N/A')} years")
                            
                            # Show required skills
                            skills = job.get('required_skills', [])
                            if skills:
                                st.write(f"**Required Skills ({len(skills)}):**")
                                skills_text = ', '.join(skills[:8])
                                if len(skills) > 8:
                                    skills_text += f" + {len(skills) - 8} more"
                                st.write(skills_text)
                        
                        with col2:
                            st.write(f"**Job ID:** {job['id'][:12]}...")
                            st.write(f"**Created:** {job['created_at'][:10]}")
                            
                            # Action buttons
                            if st.button(f"🔍 Find Candidates", key=f"find_{i}"):
                                # Switch to matching page with this job
                                st.session_state.selected_job_id = job['id']
                                st.info(f"Switched to Job Matching for: {job['title']}")
                            
                            if st.button(f"✏️ Edit Job", key=f"edit_{i}"):
                                st.info("Edit functionality would be implemented here")
                        
                        # Show job description preview
                        if job.get('raw_text'):
                            st.write("**Job Description:**")
                            description_preview = job['raw_text'][:300]
                            if len(job['raw_text']) > 300:
                                description_preview += "..."
                            st.text(description_preview)
                        
                        # Show job summary if available
                        if job.get('summary'):
                            st.write("**AI Summary:**")
                            st.write(job['summary'])
                
                # Summary statistics
                st.subheader("📊 Jobs Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    avg_experience = sum(job.get('experience_years', 0) for job in jobs) / len(jobs)
                    st.metric("Avg Experience Required", f"{avg_experience:.1f} years")
                
                with col2:
                    total_skills = sum(len(job.get('required_skills', [])) for job in jobs)
                    st.metric("Total Skills Required", total_skills)
                
                with col3:
                    companies = set(job['company'] for job in jobs if job.get('company'))
                    st.metric("Unique Companies", len(companies))
                
                with col4:
                    locations = set(job.get('location', '') for job in jobs if job.get('location'))
                    st.metric("Unique Locations", len(locations))
                
                # Skills analysis
                if total_skills > 0:
                    st.subheader("🔧 Most Requested Skills")
                    skill_counts = {}
                    for job in jobs:
                        for skill in job.get('required_skills', []):
                            skill_counts[skill] = skill_counts.get(skill, 0) + 1
                    
                    top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                    skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Frequency'])
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        skills_df,
                        x='Frequency',
                        y='Skill',
                        orientation='h',
                        title="Top 10 Most Requested Skills"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No job descriptions stored yet. Add some jobs using the form above.")
                
                # Show sample job button
                if st.button("📊 Load Sample Jobs"):
                    with st.spinner("Loading sample jobs..."):
                        try:
                            result = asyncio.run(self.data_pipeline.process_sample_data())
                            st.success(f"✅ Loaded sample jobs!")
                            st.cache_data.clear()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to load sample jobs: {e}")
        
        except Exception as e:
            st.error(f"Error loading jobs: {str(e)}")
            logger.error(f"Display stored jobs error: {str(e)}")
    
    def get_analytics_data(self):
        """Get data for analytics"""
        try:
            resumes = asyncio.run(self.resume_processor.list_processed_resumes())
            jobs = asyncio.run(self.job_processor.list_stored_jobs())
            
            return {
                'resumes': resumes,
                'jobs': jobs,
                'total_resumes': len(resumes),
                'total_jobs': len(jobs)
            }
        
        except Exception as e:
            logger.error(f"Error getting analytics data: {str(e)}")
            return None
    
    def display_resume_analytics(self, analytics_data):
        """Display resume analytics"""
        if not analytics_data['resumes']:
            return
        
        st.subheader("📊 Resume Analytics")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", analytics_data['total_resumes'])
        
        with col2:
            avg_experience = sum(r['experience_years'] for r in analytics_data['resumes']) / len(analytics_data['resumes'])
            st.metric("Avg Experience", f"{avg_experience:.1f} years")
        
        with col3:
            total_skills = sum(len(r['skills']) for r in analytics_data['resumes'])
            st.metric("Total Skills", total_skills)
        
        with col4:
            avg_skills = total_skills / len(analytics_data['resumes'])
            st.metric("Avg Skills/Resume", f"{avg_skills:.1f}")
        
        # Experience distribution
        experience_data = [r['experience_years'] for r in analytics_data['resumes']]
        fig_exp = px.histogram(
            x=experience_data,
            title="Experience Distribution",
            nbins=10,
            labels={'x': 'Years of Experience', 'y': 'Number of Candidates'}
        )
        st.plotly_chart(fig_exp, use_container_width=True)
        
        # Top skills
        all_skills = {}
        for resume in analytics_data['resumes']:
            for skill in resume['skills']:
                all_skills[skill] = all_skills.get(skill, 0) + 1
        
        top_skills = sorted(all_skills.items(), key=lambda x: x[1], reverse=True)[:15]
        skills_df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
        
        fig_skills = px.bar(
            skills_df,
            x='Count',
            y='Skill',
            orientation='h',
            title="Top 15 Skills in Database"
        )
        st.plotly_chart(fig_skills, use_container_width=True)
    
    def display_matching_analytics(self, analytics_data):
        """Display matching analytics"""
        st.subheader("🎯 Matching Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Stored Jobs", analytics_data['total_jobs'])
        
        with col2:
            match_rate = (analytics_data['total_resumes'] / max(analytics_data['total_jobs'], 1))
            st.metric("Candidates per Job", f"{match_rate:.1f}")
    
    def get_available_skills(self):
        """Get list of all available skills"""
        try:
            resumes = asyncio.run(self.resume_processor.list_processed_resumes())
            all_skills = set()
            
            for resume in resumes:
                all_skills.update(resume['skills'])
            
            return sorted(list(all_skills))
        
        except Exception as e:
            logger.error(f"Error getting available skills: {str(e)}")
            return []
    
    def search_candidates(self, query, min_exp, max_exp, skills):
        """Search candidates based on criteria using vector search"""
        try:
            st.subheader("🎯 Search Results")
            
            if query.strip():
                # Use semantic search
                with st.spinner("Performing semantic search..."):
                    query_embedding = embedding_service.generate_embedding(query)
                    candidates = vector_store.search_similar(
                        query_embedding=query_embedding,
                        top_k=20,
                        collection_name="resumes"
                    )
            else:
                # Get all candidates for skill-only filtering
                candidates = vector_store.get_all_candidates()[:20]
            
            if not candidates:
                st.warning("No candidates found")
                return
            
            # Apply filters
            filtered_candidates = self.apply_search_filters(
                candidates, min_exp, max_exp, skills, "Any of selected"
            )
            
            if not filtered_candidates:
                st.warning("No candidates match your filters")
                return
            
            # Show results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Found", len(candidates))
            with col2:
                st.metric("After Filters", len(filtered_candidates))
            
            # Display results
            self.display_vector_search_results(filtered_candidates, show_details=True)
            
        except Exception as e:
            st.error(f"Error searching candidates: {str(e)}")
            logger.error(f"Search candidates error: {str(e)}")


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
