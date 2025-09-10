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
                ["📄 Resume Upload", "📋 Job Management", "🎯 Job Matching", "📊 Analytics", "� Data Pipeline"]
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
        stored_jobs = asyncio.run(self.job_processor.list_stored_jobs())
        
        if not stored_jobs:
            st.warning("No job descriptions stored. Please add jobs in the Job Management page first.")
            return
        
        # Job selection
        job_options = {f"{job['title']} - {job['company']}": job['id'] for job in stored_jobs}
        selected_job_name = st.selectbox("Select Job for Matching", list(job_options.keys()))
        
        if selected_job_name:
            job_id = job_options[selected_job_name]
            
            col1, col2 = st.columns([2, 1])
            with col1:
                top_k = st.slider("Number of top candidates", min_value=5, max_value=50, value=10)
            with col2:
                auto_refresh = st.checkbox("Auto-refresh results", value=False)
            
            if st.button("🔍 Find Matches", type="primary") or auto_refresh:
                self.perform_job_matching(job_id, top_k)
    
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
    
    def search_page(self):
        """Search and filter page"""
        st.header("🔍 Search & Filter Candidates")
        
        # Search options
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            search_query = st.text_input("Search candidates", placeholder="Enter skills, experience, or keywords...")
        
        with col2:
            min_experience = st.number_input("Min Experience (years)", min_value=0, max_value=20, value=0)
        
        with col3:
            max_experience = st.number_input("Max Experience (years)", min_value=0, max_value=20, value=20)
        
        # Skill filters
        available_skills = self.get_available_skills()
        selected_skills = st.multiselect("Filter by Skills", available_skills)
        
        if st.button("🔎 Search Candidates"):
            self.search_candidates(search_query, min_experience, max_experience, selected_skills)
    
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
    
    def perform_job_matching(self, job_id, top_k):
        """Perform job matching and display results"""
        try:
            # Get job data
            job_data = asyncio.run(self.job_processor.get_job_data(job_id))
            if not job_data:
                st.error("Job not found")
                return
            
            st.info(f"🔍 Finding matches for: {job_data.title}")
            
            # Find matches
            matches = asyncio.run(self.resume_processor.find_best_matches(job_data, top_k))
            
            if not matches:
                st.warning("No matching candidates found")
                return
            
            # Display results
            st.subheader(f"🎯 Top {len(matches)} Candidates")
            
            # Create visualization
            self.visualize_match_results(matches)
            
            # Display detailed results
            self.display_match_results(matches)
            
        except Exception as e:
            st.error(f"Error performing matching: {str(e)}")
    
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
        """Display list of stored jobs"""
        try:
            jobs = asyncio.run(self.job_processor.list_stored_jobs())
            
            if jobs:
                st.subheader(f"📋 Stored Jobs ({len(jobs)})")
                
                # Create dataframe for display
                df = pd.DataFrame([{
                    'Title': job['title'],
                    'Company': job['company'],
                    'Experience': f"{job.get('experience_years', 'N/A')} years",
                    'Stored': job['created_at'][:10]  # Date only
                } for job in jobs])
                
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No job descriptions stored yet.")
        
        except Exception as e:
            st.error(f"Error loading jobs: {str(e)}")
    
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
        """Search candidates based on criteria"""
        try:
            # This would be implemented with vector search in the real system
            st.info("Search functionality would be implemented here")
            
        except Exception as e:
            st.error(f"Error searching candidates: {str(e)}")


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
