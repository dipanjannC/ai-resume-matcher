#!/usr/bin/env python3
"""
Integration test script for AI Resume Matcher
Tests key functionality including improved LangChain parsing and data pipeline
"""

import asyncio
import sys
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_integration():
    """Test the core integration features with comprehensive validation"""
    
    print("🧪 Running AI Resume Matcher Integration Tests")
    print("=" * 60)
    
    test_results = {
        "passed": 0,
        "failed": 0,
        "warnings": 0
    }
    
    try:
        # Test 1: Import all core modules
        print("1. Testing module imports...")
        from app.services.resume_processor import ResumeProcessor
        from app.services.job_processor import JobProcessor
        from app.services.data_pipeline import DataPipeline
        from app.models.resume_data import ResumeData, JobDescription, MatchResult
        from app.services.langchain_agents import langchain_agents
        print("   ✅ All core modules imported successfully")
        test_results["passed"] += 1
        
        # Test 2: Initialize services
        print("\n2. Testing service initialization...")
        resume_processor = ResumeProcessor()
        job_processor = JobProcessor()
        data_pipeline = DataPipeline()
        print("   ✅ All services initialized successfully")
        test_results["passed"] += 1
        
        # Test 3: Validate sample data availability
        print("\n3. Validating sample data...")
        csv_file = Path("data/samples/job_title_des.csv")
        resume_dir = Path("data/processed_resumes")
        
        if csv_file.exists():
            df = pd.read_csv(csv_file)
            print(f"   ✅ Sample CSV found: {len(df)} job descriptions available")
            print(f"   📋 CSV columns: {list(df.columns)}")
            test_results["passed"] += 1
        else:
            print(f"   ❌ Sample CSV not found: {csv_file}")
            test_results["failed"] += 1
        
        if resume_dir.exists():
            resume_files = list(resume_dir.glob("*.txt"))
            print(f"   ✅ Resume directory found: {len(resume_files)} resume files")
            test_results["passed"] += 1
        else:
            print(f"   ⚠️  Resume directory not found: {resume_dir}")
            test_results["warnings"] += 1
        
        # Test 4: Test LangChain job parsing
        print("\n4. Testing LangChain job parsing...")
        try:
            test_job_text = """
            Job Title: Senior Python Developer
            Company: TechCorp Inc.
            
            We are seeking a Senior Python Developer with 5+ years of experience.
            Required skills: Python, Django, FastAPI, PostgreSQL, AWS
            Preferred skills: React, Docker, Kubernetes
            Experience: 5 years minimum
            Education: Bachelor's degree in Computer Science or equivalent
            """
            
            job_result = await langchain_agents.parse_job_description(test_job_text)
            print(f"   ✅ Job parsing successful: {job_result.title} at {job_result.company}")
            print(f"   🎯 Required skills: {job_result.required_skills}")
            print(f"   📅 Experience: {job_result.experience_years} years")
            test_results["passed"] += 1
        except Exception as e:
            print(f"   ❌ Job parsing failed: {str(e)}")
            test_results["failed"] += 1
        
        # Test 5: Test data pipeline sample loading
        print("\n5. Testing data pipeline with sample data...")
        try:
            print("   📤 Processing sample data (limited to 3 jobs for speed)...")
            # Temporarily limit processing for test speed
            original_csv = Path("data/samples/job_title_des.csv")
            test_csv = Path("data/temp/test_jobs.csv")
            test_csv.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a small test dataset
            if original_csv.exists():
                df = pd.read_csv(original_csv, nrows=3)  # Only first 3 jobs
                df.to_csv(test_csv, index=False)
                
                sample_results = await data_pipeline.bulk_upload_jobs_from_csv(test_csv)
                print(f"   ✅ Sample data processing: {sample_results['processed']} jobs processed")
                print(f"   📊 Processing stats: {sample_results['failed']} failed")
                
                # Clean up test file
                if test_csv.exists():
                    test_csv.unlink()
                    
                test_results["passed"] += 1
            else:
                print("   ⚠️  Skipping data pipeline test - no sample CSV")
                test_results["warnings"] += 1
                
        except Exception as e:
            print(f"   ❌ Data pipeline test failed: {str(e)}")
            test_results["failed"] += 1
        
        # Test 6: Test job listing and storage
        print("\n6. Testing job management...")
        try:
            jobs = await job_processor.list_stored_jobs()
            print(f"   ✅ Jobs in database: {len(jobs)} jobs")
            
            if jobs:
                # Display info about stored jobs
                job_list = jobs[:3]  # Show first 3
                for i, job in enumerate(job_list, 1):
                    print(f"   📝 Job {i}: {job['title']} at {job['company']}")
            
            test_results["passed"] += 1
        except Exception as e:
            print(f"   ❌ Job management test failed: {str(e)}")
            test_results["failed"] += 1
        
        # Test 7: Test candidate matching (if jobs and resumes exist)
        print("\n7. Testing candidate matching...")
        try:
            jobs = await job_processor.list_stored_jobs()
            if jobs:
                job_id = jobs[0]['id']  # Get first job's ID
                job_title = jobs[0]['title']
                print(f"   🔍 Testing candidate search for: {job_title}")
                
                candidates = await job_processor.find_candidates_for_job(job_id, top_k=3)
                print(f"   ✅ Found {len(candidates)} candidate matches")
                
                if candidates:
                    top_candidate = candidates[0]
                    print(f"   🏆 Top candidate score: {top_candidate['similarity_score']:.3f}")
                    print(f"   👤 Candidate ID: {top_candidate['candidate_id']}")
                
                test_results["passed"] += 1
            else:
                print("   ⚠️  Skipping candidate matching (no jobs in database)")
                test_results["warnings"] += 1
                
        except Exception as e:
            print(f"   ❌ Candidate matching test failed: {str(e)}")
            test_results["failed"] += 1
        
        # Test 8: Test data models and validation
        print("\n8. Testing data models...")
        try:
            test_match = MatchResult(
                resume_id="test-resume-123",
                job_id="test-job-456",
                overall_score=0.85,
                skills_match_score=0.9,
                experience_match_score=0.8,
                semantic_similarity_score=0.85,
                candidate_name="Test Candidate"
            )
            print(f"   ✅ MatchResult model: {test_match.candidate_name} - Score: {test_match.overall_score}")
            
            # Test serialization
            match_dict = test_match.to_dict()
            print(f"   ✅ Model serialization: {len(match_dict)} fields")
            
            test_results["passed"] += 1
        except Exception as e:
            print(f"   ❌ Data model test failed: {str(e)}")
            test_results["failed"] += 1
        
        # Test 9: Test pipeline statistics
        print("\n9. Testing pipeline statistics...")
        try:
            stats = data_pipeline.get_pipeline_stats()
            print(f"   ✅ Pipeline stats: {stats['total_jobs']} jobs, {stats['total_resumes']} resumes")
            test_results["passed"] += 1
        except Exception as e:
            print(f"   ❌ Pipeline stats test failed: {str(e)}")
            test_results["failed"] += 1
        
        # Test Summary
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print(f"   ✅ Passed: {test_results['passed']}")
        print(f"   ❌ Failed: {test_results['failed']}")
        print(f"   ⚠️  Warnings: {test_results['warnings']}")
        
        if test_results["failed"] == 0:
            print("\n🎉 All critical tests PASSED!")
            print("   The AI Resume Matcher is ready for production use.")
            print("\n🚀 Launch commands:")
            print("   Streamlit App: streamlit run streamlit_app.py --server.port 8503")
            print("   Or with venv: ./venv/bin/streamlit run streamlit_app.py --server.port 8503")
            return True
        else:
            print(f"\n⚠️  Some tests failed ({test_results['failed']} failures)")
            print("   Please review the errors above before deployment.")
            return False
        
    except Exception as e:
        print(f"\n❌ Integration test failed with critical error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_integration())
    sys.exit(0 if success else 1)
