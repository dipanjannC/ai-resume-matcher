"""
Simple main entry point for the AI Resume Matcher.
Provides both programmatic API and CLI interface.
"""

import asyncio
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path

from app.services.resume_processor import resume_processor
from app.models.resume_data import JobMatchRequest, MatchResponse
from app.utils.file_utils import save_uploaded_file, cleanup_temp_file, validate_file_type, validate_file_size
from app.core.config import settings
from app.core.logging import get_logger
from app.core.exceptions import ResumeMatcherException

logger = get_logger(__name__)

# FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered resume matching using LangChain agents"
)


@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None)
):
    """Upload and process a resume file"""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(400, "No filename provided")

        if not validate_file_type(file.filename, settings.ALLOWED_FILE_TYPES.split(",")):
            raise HTTPException(400, f"Unsupported file type. Allowed: {settings.ALLOWED_FILE_TYPES}")
        
        content = await file.read()
        if not validate_file_size(content, settings.MAX_FILE_SIZE_MB):
            raise HTTPException(400, f"File too large. Max size: {settings.MAX_FILE_SIZE_MB}MB")
        
        # Save temporary file and process
        temp_path = save_uploaded_file(content, file.filename)
        try:
            resume_data = await resume_processor.process_resume_file(temp_path, file.filename)
            return JSONResponse({
                "success": True,
                "resume_id": resume_data.id,
                "filename": resume_data.filename,
                "summary": resume_data.to_dict()
            })
        finally:
            cleanup_temp_file(temp_path)
            
    except ResumeMatcherException as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(500, "Internal server error")


@app.post("/match-job", response_model=MatchResponse)
async def match_job(request: JobMatchRequest):
    """Find best resume matches for a job description"""
    try:
        # Process job description
        job_data = await resume_processor.process_job_description(
            request.job_description, 
            request.job_title or "", 
            request.company or ""
        )
        
        # Find matches
        matches = await resume_processor.find_best_matches(job_data, top_k=10)
        
        # Format response
        match_results = []
        for match in matches:
            resume_data = await resume_processor._get_resume_data(match.resume_id)
            match_results.append({
                "resume_id": match.resume_id,
                "candidate_name": resume_data.profile.name if resume_data else "Unknown",
                "overall_score": match.overall_score,
                "skills_match": match.skills_match_score,
                "experience_match": match.experience_match_score,
                "matching_skills": match.matching_skills,
                "missing_skills": match.missing_skills,
                "recommendation": match.recommendation
            })
        
        return MatchResponse(
            matches=match_results,
            total_candidates=len(await resume_processor.list_processed_resumes()),
            processing_time=0.0  # Could add timing if needed
        )
        
    except ResumeMatcherException as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Matching error: {str(e)}")
        raise HTTPException(500, "Internal server error")


@app.get("/resumes")
async def list_resumes():
    """List all processed resumes"""
    try:
        resumes = await resume_processor.list_processed_resumes()
        return JSONResponse({"resumes": resumes})
    except Exception as e:
        logger.error(f"List resumes error: {str(e)}")
        raise HTTPException(500, "Internal server error")


@app.get("/resumes/{resume_id}")
async def get_resume(resume_id: str):
    """Get detailed resume analysis"""
    try:
        summary = await resume_processor.get_resume_summary(resume_id)
        return JSONResponse(summary)
    except ResumeMatcherException as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        logger.error(f"Get resume error: {str(e)}")
        raise HTTPException(500, "Internal server error")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": settings.APP_VERSION}


# CLI mode
async def cli_mode():
    """Run in CLI mode"""
    print(f"🤖 {settings.APP_NAME} v{settings.APP_VERSION}")
    print("Starting in CLI mode...")
    
    # Import and run demo
    from demo_example import main as demo_main
    await demo_main()


def main():
    """Main entry point"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        # Run in CLI mode
        asyncio.run(cli_mode())
    else:
        # Run as web server
        print(f"🚀 Starting {settings.APP_NAME} v{settings.APP_VERSION}")
        print(f"📊 Data directory: {settings.DATA_DIR}")
        print(f"🔗 OpenAI configured: {'Yes' if settings.OPENAI_API_KEY else 'No'}")
        
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=settings.DEBUG,
            log_level="info"
        )


if __name__ == "__main__":
    main()
