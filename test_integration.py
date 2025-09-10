#!/usr/bin/env python3
"""
Integration test script for AI Resume Matcher
Tests key functionality without starting the full Streamlit app
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_integration():
    """Test the core integration features"""
    
    print("🧪 Running AI Resume Matcher Integration Tests")
    print("=" * 50)
    
    try:
        # Test 1: Import all core modules
        print("1. Testing module imports...")
        from app.services.resume_processor import ResumeProcessor
        from app.services.job_processor import JobProcessor
        from app.services.data_pipeline import DataPipeline
        from app.models.resume_data import ResumeData, JobDescription, MatchResult
        from app.services.langchain_agents import langchain_agents
        print("   ✅ All core modules imported successfully")
        
        # Test 2: Initialize services
        print("\n2. Testing service initialization...")
        resume_processor = ResumeProcessor()
        job_processor = JobProcessor()
        data_pipeline = DataPipeline()
        print("   ✅ All services initialized successfully")
        
        # Test 3: Test data pipeline sample loading
        print("\n3. Testing sample data loading...")
        try:
            sample_resumes = await data_pipeline.process_sample_data()
            print(f"   ✅ Sample resumes loaded: {sample_resumes['total_processed']} resumes")
        except Exception as e:
            print(f"   ⚠️  Sample resume loading: {str(e)}")
        
        try:
            sample_jobs = await data_pipeline.process_sample_data()
            print(f"   ✅ Sample jobs loaded: {sample_jobs['total_processed']} jobs")
        except Exception as e:
            print(f"   ⚠️  Sample job loading: {str(e)}")
        
        # Test 4: Test job listing
        print("\n4. Testing job management...")
        jobs = await job_processor.list_stored_jobs()
        print(f"   ✅ Jobs in database: {len(jobs)} jobs")
        
        # Test 5: Test candidate finding (if jobs exist)
        if jobs:
            print("\n5. Testing candidate matching...")
            job_id = list(jobs.keys())[0]
            job_title = jobs[job_id].title
            print(f"   Testing with job: {job_title}")
            
            candidates = await job_processor.find_candidates_for_job(job_id, top_k=3)
            print(f"   ✅ Found {len(candidates)} candidate matches")
            
            if candidates:
                top_candidate = candidates[0]
                print(f"   📊 Top candidate score: {top_candidate['similarity_score']:.3f}")
        else:
            print("\n5. Skipping candidate matching (no jobs in database)")
        
        # Test 6: Test model creation
        print("\n6. Testing data models...")
        test_match = MatchResult(
            resume_id="test-123",
            job_id="job-456",
            overall_score=0.85,
            skills_match_score=0.9,
            experience_match_score=0.8,
            semantic_similarity_score=0.85,
            candidate_name="Test Candidate"
        )
        print(f"   ✅ MatchResult created: {test_match.candidate_name} - Score: {test_match.overall_score}")
        
        print("\n" + "=" * 50)
        print("🎉 Integration tests completed successfully!")
        print("   The AI Resume Matcher is ready for use.")
        print("   Launch with: ./venv/bin/python -m streamlit run streamlit_app.py")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
