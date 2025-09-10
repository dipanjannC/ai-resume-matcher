#!/usr/bin/env python3
"""Test with sample data to verify all fixes work."""

import asyncio
import sys
import os

sys.path.append(os.path.abspath('.'))

from app.services.data_pipeline import DataPipeline
from pathlib import Path

async def test_with_sample_data():
    """Test processing with actual sample CSV data."""
    print("🧪 Testing with Sample CSV Data")
    print("=" * 40)
    
    try:
        pipeline = DataPipeline()
        csv_path = Path('data/samples/job_title_des.csv')
        
        if not csv_path.exists():
            print("❌ Sample CSV not found")
            return False
            
        print("Testing with sample CSV...")
        result = await pipeline.bulk_upload_jobs_from_csv(csv_path)
        
        print(f"Processed: {result['processed']}")
        print(f"Failed: {result['failed']}")
        
        if result['errors']:
            print("Errors encountered:")
            for error in result['errors']:
                print(f"  - {error}")
        
        if result['failed'] == 0:
            print('✅ All jobs processed successfully!')
            return True
        else:
            print(f'❌ {result["failed"]} jobs failed')
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_with_sample_data())
    if success:
        print("\n🎉 All metadata and parsing fixes working!")
    else:
        print("\n💥 Still have issues to resolve")
        sys.exit(1)
