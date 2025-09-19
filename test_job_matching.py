#!/usr/bin/env python3
"""
Test job matching functionality
"""

import sys
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.job_processor import job_processor
from app.services.vector_store import vector_store

async def test_job_matching():
    """Test the job matching functionality"""
    
    print("🎯 Testing Job Matching Functionality")
    print("=" * 50)
    
    try:
        # Check if we have any jobs
        jobs = await job_processor.list_stored_jobs()
        print(f"📋 Found {len(jobs)} stored jobs")
        
        if len(jobs) == 0:
            print("⚠️  No jobs found. Creating a test job...")
            
            # Create a test job
            test_job = await job_processor.process_and_store_job(
                job_text="We are looking for a Senior Python Developer with API Gateway experience. Must have Kong or Apigee experience, microservices architecture knowledge, and 5+ years of experience.",
                title="Senior Python Developer",
                company="TechCorp",
                experience_years=5,
                location="San Francisco, CA"
            )
            print(f"✅ Created test job: {test_job.title}")
            jobs = [test_job.to_dict()]
        
        # Check resume data
        candidates = vector_store.get_all_candidates()
        print(f"👥 Found {len(candidates)} resume candidates")
        
        if len(candidates) == 0:
            print("⚠️  No resumes found. Please load sample data first.")
            return
        
        # Test job matching with first job
        test_job = jobs[0]
        job_id = test_job['id']
        
        print(f"\n🔍 Testing matching for job: {test_job['title']}")
        
        # Find candidates for this job
        matches = await job_processor.find_candidates_for_job(job_id, top_k=5)
        
        print(f"🎯 Found {len(matches)} matching candidates:")
        
        for i, match in enumerate(matches, 1):
            similarity = match.get('similarity_score', match.get('similarity', 0))
            candidate_id = match.get('candidate_id', 'Unknown')
            metadata = match.get('metadata', {})
            name = metadata.get('name', 'Unknown')
            
            print(f"   {i}. {name} (ID: {candidate_id[:12]}...) - Score: {similarity:.3f}")
        
        if len(matches) > 0:
            print("\n✅ Job matching is working correctly!")
        else:
            print("\n❌ Job matching returned no results - there may be an issue")
        
    except Exception as e:
        print(f"❌ Error testing job matching: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_job_matching())
