"""
Data Pipeline - Bulk upload and processing for resumes and job descriptions
"""

import asyncio
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

from app.services.resume_processor import resume_processor
from app.services.job_processor import job_processor
from app.core.logging import get_logger
from app.core.exceptions import ResumeMatcherException

logger = get_logger(__name__)


class DataPipeline:
    """
    Data pipeline for bulk processing of resumes and job descriptions
    """
    
    def __init__(self):
        self.resume_processor = resume_processor
        self.job_processor = job_processor
        self.data_dir = Path("data")
        self.batch_size = 10
    
    async def bulk_upload_resumes(self, resume_files: List[Path], progress_callback=None) -> Dict[str, Any]:
        """
        Bulk upload and process resume files using content processing
        
        Args:
            resume_files: List of resume file paths
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results summary
        """
        results = {
            "total_files": len(resume_files),
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        logger.info(f"Starting bulk resume upload: {len(resume_files)} files")
        
        for i, file_path in enumerate(resume_files):
            try:
                if progress_callback:
                    progress_callback(i + 1, len(resume_files), f"Processing {file_path.name}")
                
                # Read resume content directly for text files
                if file_path.suffix.lower() == '.txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if content:
                        # Process resume content using resume_processor
                        resume_data = await self.resume_processor.process_resume_content(
                            content=content,
                            filename=file_path.name
                        )
                        results["processed"] += 1
                        logger.info(f"Processed resume: {file_path.name} -> {resume_data.id}")
                    else:
                        logger.warning(f"Empty resume file: {file_path.name}")
                        results["failed"] += 1
                else:
                    # For other file types, use file processing
                    resume_data = await self.resume_processor.process_resume_file(str(file_path))
                    results["processed"] += 1
                    logger.info(f"Processed resume: {file_path.name} -> {resume_data.id}")
                
            except Exception as e:
                error_msg = f"Failed to process {file_path.name}: {str(e)}"
                results["errors"].append(error_msg)
                results["failed"] += 1
                logger.error(error_msg)
        
        logger.info(f"Bulk resume upload completed: {results['processed']} processed, {results['failed']} failed")
        return results
    
    async def bulk_upload_jobs_from_csv(self, csv_file: Path, progress_callback=None) -> Dict[str, Any]:
        """
        Bulk upload job descriptions from CSV file
        
        Expected CSV columns: "Job Title", "Job Description" (matches sample CSV format)
        
        Args:
            csv_file: Path to CSV file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results summary
        """
        results = {
            "total_jobs": 0,
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Read CSV file
            df = pd.read_csv(csv_file)
            results["total_jobs"] = len(df)
            
            logger.info(f"Starting bulk job upload from CSV: {len(df)} jobs")
            
            # Map CSV columns to expected format
            column_mapping = {
                'Job Title': 'title',
                'Job Description': 'description',
                'title': 'title',  # Already correct
                'description': 'description'  # Already correct
            }
            
            # Check for required columns
            title_col = None
            desc_col = None
            
            for col in df.columns:
                if col in ['Job Title', 'title']:
                    title_col = col
                elif col in ['Job Description', 'description']:
                    desc_col = col
            
            if not title_col or not desc_col:
                raise ResumeMatcherException(f"Missing required columns. Expected 'Job Title' and 'Job Description' or 'title' and 'description'. Found: {list(df.columns)}")
            
            # Process each job
            for i in range(len(df)):
                try:
                    row = df.iloc[i]
                    if progress_callback:
                        progress_callback(i + 1, len(df), f"Processing job: {row.get(title_col, 'Unknown')}")
                    
                    # Extract job data with safe string conversion
                    title = str(row.get(title_col, '')).strip()
                    description = str(row.get(desc_col, '')).strip()
                    
                    # Skip empty rows
                    if not title or not description or title == 'nan' or description == 'nan':
                        logger.warning(f"Skipping empty job at row {i}")
                        continue
                    
                    # Extract additional data if available
                    company = str(row.get('company', row.get('Company', ''))).strip()
                    experience_years = 0
                    location = str(row.get('location', row.get('Location', ''))).strip()
                    
                    # Try to parse experience years
                    try:
                        exp_val = row.get('experience_years', row.get('Experience Years', 0))
                        if pd.notna(exp_val):
                            experience_years = int(float(exp_val))
                    except (ValueError, TypeError):
                        experience_years = 0
                    
                    # Process and store job using job_processor
                    processed_job = await self.job_processor.process_and_store_job(
                        job_text=description,
                        title=title,
                        company=company if company and company != 'nan' else "",
                        experience_years=experience_years if experience_years > 0 else 0,
                        location=location if location and location != 'nan' else ""
                    )
                    
                    results["processed"] += 1
                    logger.info(f"Processed job: {title} -> {processed_job.id}")
                    
                except Exception as e:
                    error_msg = f"Failed to process job row {i}: {str(e)}"
                    results["errors"].append(error_msg)
                    results["failed"] += 1
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to read CSV file {csv_file}: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        logger.info(f"Bulk job upload completed: {results['processed']} processed, {results['failed']} failed")
        return results
    
    async def bulk_upload_jobs_from_json(self, json_file: Path, progress_callback=None) -> Dict[str, Any]:
        """
        Bulk upload job descriptions from JSON file
        
        Expected JSON format: List of job objects with title, company, description, etc.
        
        Args:
            json_file: Path to JSON file
            progress_callback: Optional callback for progress updates
            
        Returns:
            Processing results summary
        """
        results = {
            "total_jobs": 0,
            "processed": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            # Read JSON file
            with open(json_file, 'r') as f:
                jobs_data = json.load(f)
            
            if not isinstance(jobs_data, list):
                jobs_data = [jobs_data]
            
            results["total_jobs"] = len(jobs_data)
            logger.info(f"Starting bulk job upload from JSON: {len(jobs_data)} jobs")
            
            # Process each job
            for i, job_data in enumerate(jobs_data):
                try:
                    if progress_callback:
                        progress_callback(i + 1, len(jobs_data), f"Processing job: {job_data.get('title', 'Unknown')}")
                    
                    # Process and store job
                    processed_job = await self.job_processor.process_and_store_job(
                        job_text=job_data.get('description', ''),
                        title=job_data.get('title', ''),
                        company=job_data.get('company', ''),
                        experience_years=job_data.get('experience_years', 0),
                        location=job_data.get('location', '')
                    )
                    
                    results["processed"] += 1
                    logger.info(f"Processed job: {job_data.get('title', 'Unknown')} -> {processed_job.id}")
                    
                except Exception as e:
                    error_msg = f"Failed to process job {i}: {str(e)}"
                    results["errors"].append(error_msg)
                    results["failed"] += 1
                    logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Failed to read JSON file {json_file}: {str(e)}"
            results["errors"].append(error_msg)
            logger.error(error_msg)
        
        logger.info(f"Bulk job upload completed: {results['processed']} processed, {results['failed']} failed")
        return results
    
    def load_sample_data(self) -> Dict[str, Any]:
        """
        Load sample data for demonstration
        
        Returns:
            Summary of loaded data
        """
        logger.info("Loading sample data...")
        
        # Create sample job descriptions
        sample_jobs = [
            {
                "title": "Senior Python Developer",
                "company": "TechCorp Inc.",
                "description": "We are seeking a Senior Python Developer with 5+ years of experience in Django, FastAPI, and cloud technologies. Must have experience with PostgreSQL, Redis, and AWS.",
                "experience_years": 5,
                "location": "San Francisco, CA"
            },
            {
                "title": "Data Scientist",
                "company": "DataTech Solutions",
                "description": "Looking for a Data Scientist with expertise in machine learning, Python, R, and SQL. Experience with TensorFlow, PyTorch, and big data tools required.",
                "experience_years": 3,
                "location": "New York, NY"
            },
            {
                "title": "Frontend Developer",
                "company": "WebDev Studio",
                "description": "Frontend Developer needed with React, JavaScript, TypeScript, and CSS expertise. Experience with Node.js and modern build tools preferred.",
                "experience_years": 2,
                "location": "Austin, TX"
            },
            {
                "title": "DevOps Engineer",
                "company": "CloudFirst",
                "description": "DevOps Engineer with Docker, Kubernetes, AWS, and CI/CD pipeline experience. Must know Infrastructure as Code and monitoring tools.",
                "experience_years": 4,
                "location": "Seattle, WA"
            },
            {
                "title": "Full Stack Developer",
                "company": "Startup Hub",
                "description": "Full Stack Developer with Python, JavaScript, React, and database experience. Startup environment with modern tech stack.",
                "experience_years": 3,
                "location": "Remote"
            }
        ]
        
        # Save to JSON file
        sample_file = self.data_dir / "samples" / "sample_jobs.json"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(sample_file, 'w') as f:
            json.dump(sample_jobs, f, indent=2)
        
        return {
            "sample_jobs_created": len(sample_jobs),
            "file_path": str(sample_file),
            "jobs": sample_jobs
        }
    
    async def process_sample_data(self) -> Dict[str, Any]:
        """
        Process and load sample data into vector store
        Uses the actual job_title_des.csv file from samples folder
        
        Returns:
            Processing results
        """
        logger.info("Processing sample data...")
        
        results = {
            "jobs": {"processed": 0, "failed": 0, "errors": []},
            "resumes": {"processed": 0, "failed": 0, "errors": []},
            "total_processed": 0
        }
        
        try:
            # Process sample jobs from CSV
            csv_file = self.data_dir / "samples" / "job_title_des.csv"
            if csv_file.exists():
                logger.info(f"Loading jobs from {csv_file}")
                # Process only first 10 jobs for demo to avoid long processing time
                df = pd.read_csv(csv_file)
                df_sample = df.head(10)  # Take first 10 jobs
                
                # Save sample to temp file for processing
                temp_csv = self.data_dir / "temp" / "sample_jobs_temp.csv"
                temp_csv.parent.mkdir(parents=True, exist_ok=True)
                df_sample.to_csv(temp_csv, index=False)
                
                # Process the sample jobs
                job_results = await self.bulk_upload_jobs_from_csv(temp_csv)
                results["jobs"] = job_results
                results["total_processed"] += job_results["processed"]
                
                # Clean up temp file
                if temp_csv.exists():
                    temp_csv.unlink()
                    
            else:
                logger.warning(f"Sample CSV file not found: {csv_file}")
                results["jobs"]["errors"].append(f"Sample CSV file not found: {csv_file}")
            
            # Also load sample resumes if they exist
            resume_dir = self.data_dir / "processed_resumes"
            if resume_dir.exists():
                resume_files = list(resume_dir.glob("*.txt"))[:5]  # Process first 5 for demo
                if resume_files:
                    logger.info(f"Loading {len(resume_files)} sample resumes")
                    resume_results = await self.bulk_upload_resumes(resume_files)
                    results["resumes"] = resume_results
                    results["total_processed"] += resume_results["processed"]
                else:
                    logger.info("No resume files found in processed_resumes directory")
            else:
                logger.info("Processed resumes directory not found")
        
        except Exception as e:
            error_msg = f"Failed to process sample data: {str(e)}"
            logger.error(error_msg)
            results["jobs"]["errors"].append(error_msg)
        
        logger.info(f"Sample data processing completed: {results['total_processed']} items processed")
        return results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """
        Get statistics about processed data
        
        Returns:
            Data pipeline statistics
        """
        try:
            # Get resume stats
            resume_stats = asyncio.run(self.resume_processor.list_processed_resumes())
            
            # Get job stats
            job_stats = asyncio.run(self.job_processor.list_stored_jobs())
            
            return {
                "total_resumes": len(resume_stats) if resume_stats else 0,
                "total_jobs": len(job_stats) if job_stats else 0,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get pipeline stats: {str(e)}")
            return {
                "total_resumes": 0,
                "total_jobs": 0,
                "error": str(e)
            }


# Global instance
data_pipeline = DataPipeline()
