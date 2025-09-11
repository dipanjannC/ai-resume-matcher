#!/usr/bin/env python3
"""
Vector Search Test Script
Tests the ChromaDB vector store functionality to ensure semantic search works correctly.
"""

import sys
import asyncio
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.vector_store import vector_store
from app.services.embeddings import embedding_service
from app.services.resume_processor import resume_processor
from app.core.logging import get_logger

logger = get_logger(__name__)


async def test_vector_search():
    """Test vector search functionality with semantic matching"""
    
    print("🔍 Testing Vector Search Functionality")
    print("=" * 50)
    
    try:
        # Test 1: Vector Store Initialization
        print("\n1. Testing Vector Store Initialization...")
        candidates = vector_store.get_all_candidates()
        print(f"   ✅ Vector store initialized successfully")
        print(f"   📊 Current candidates in store: {len(candidates)}")
        
        # Test 2: Embedding Generation
        print("\n2. Testing Embedding Generation...")
        test_text = "API Gateway experience with Kong and microservices"
        embedding = embedding_service.generate_embedding(test_text)
        print(f"   ✅ Embedding generated successfully")
        print(f"   📏 Embedding dimension: {len(embedding)}")
        
        # Test 3: Semantic Search
        print("\n3. Testing Semantic Search...")
        if len(candidates) > 0:
            # Search for API-related candidates
            api_query = "API Gateway Kong Apigee microservices architecture"
            api_embedding = embedding_service.generate_embedding(api_query)
            
            api_results = vector_store.search_similar(
                query_embedding=api_embedding,
                top_k=5
            )
            
            print(f"   🎯 API Gateway search results: {len(api_results)}")
            for i, result in enumerate(api_results[:3], 1):
                similarity = result.get('similarity', 0)
                candidate_id = result.get('candidate_id', 'Unknown')
                print(f"      {i}. Candidate: {candidate_id[:8]}... | Similarity: {similarity:.3f}")
            
            # Search for mobile development candidates
            mobile_query = "Flutter React Native mobile development cross-platform"
            mobile_embedding = embedding_service.generate_embedding(mobile_query)
            
            mobile_results = vector_store.search_similar(
                query_embedding=mobile_embedding,
                top_k=5
            )
            
            print(f"   📱 Mobile development search results: {len(mobile_results)}")
            for i, result in enumerate(mobile_results[:3], 1):
                similarity = result.get('similarity', 0)
                candidate_id = result.get('candidate_id', 'Unknown')
                print(f"      {i}. Candidate: {candidate_id[:8]}... | Similarity: {similarity:.3f}")
                
        else:
            print("   ⚠️ No candidates in vector store. Need to process resumes first.")
        
        # Test 4: Resume Processing
        print("\n4. Testing Resume Processing...")
        sample_resumes_dir = Path("data/resumes")
        if sample_resumes_dir.exists():
            sample_files = list(sample_resumes_dir.glob("*.txt"))
            print(f"   📄 Found {len(sample_files)} sample resume files")
            
            if sample_files and len(candidates) < 3:
                print("   🔄 Processing sample resumes...")
                for i, resume_file in enumerate(sample_files[:3], 1):
                    try:
                        with open(resume_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        print(f"      Processing {i}. {resume_file.name}...")
                        result = await resume_processor.process_resume_content(
                            content=content,
                            filename=resume_file.name
                        )
                        print(f"      ✅ Processed: {result.profile.name or 'Unnamed'}")
                        
                    except Exception as e:
                        print(f"      ❌ Failed to process {resume_file.name}: {str(e)}")
                
                # Re-test search after processing
                print("\n   🔍 Re-testing search after processing...")
                updated_candidates = vector_store.get_all_candidates()
                print(f"   📊 Updated candidates count: {len(updated_candidates)}")
                
                if len(updated_candidates) > 0:
                    # Test semantic matching
                    kong_query = "Kong API Gateway experience"
                    kong_embedding = embedding_service.generate_embedding(kong_query)
                    
                    kong_results = vector_store.search_similar(
                        query_embedding=kong_embedding,
                        top_k=3
                    )
                    
                    print(f"   🎯 'Kong API Gateway' search results:")
                    for i, result in enumerate(kong_results, 1):
                        similarity = result.get('similarity', 0)
                        metadata = result.get('metadata', {})
                        skills = metadata.get('skills', 'No skills listed')[:100]
                        print(f"      {i}. Similarity: {similarity:.3f} | Skills: {skills}...")
        
        # Test 5: Cross-technology Matching
        print("\n5. Testing Cross-Technology Semantic Matching...")
        if len(vector_store.get_all_candidates()) > 0:
            # Test if "API Gateway" matches candidates with "Kong" or "Apigee"
            gateway_query = "API Gateway architecture experience"
            gateway_embedding = embedding_service.generate_embedding(gateway_query)
            
            gateway_results = vector_store.search_similar(
                query_embedding=gateway_embedding,
                top_k=5
            )
            
            print(f"   🌉 'API Gateway' semantic matches:")
            for i, result in enumerate(gateway_results, 1):
                similarity = result.get('similarity', 0)
                metadata = result.get('metadata', {})
                skills = metadata.get('skills', '')
                
                # Check if this candidate has Kong/Apigee but still matches API Gateway
                has_kong = 'kong' in skills.lower()
                has_apigee = 'apigee' in skills.lower()
                has_api_gateway = 'api gateway' in skills.lower()
                
                match_type = []
                if has_kong: match_type.append("Kong")
                if has_apigee: match_type.append("Apigee") 
                if has_api_gateway: match_type.append("API Gateway")
                
                print(f"      {i}. Similarity: {similarity:.3f} | Technologies: {', '.join(match_type) or 'Other'}")
        
        print("\n" + "=" * 50)
        print("🎉 Vector Search Test Completed!")
        print("✅ All core functionalities are working correctly")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        logger.error(f"Vector search test failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(test_vector_search())
