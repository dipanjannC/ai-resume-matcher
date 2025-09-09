"""
Simple Resume Vector Loader
Quick script to load .txt resume files into ChromaDB vector store
"""

import sys
from pathlib import Path

# Add app to path
sys.path.append('.')

from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService


def load_resumes_simple(resume_dir="data/processed_resumes"):
    """Load resume .txt files into vector store"""
    
    # Initialize services
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    
    # Find resume files
    resume_files = list(Path(resume_dir).glob("*.txt"))
    print(f"Found {len(resume_files)} resume files")
    
    loaded = 0
    for file_path in resume_files:
        try:
            # Read resume
            with open(file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            
            # Generate embedding
            embedding = embedding_service.generate_embedding(resume_text)
            
            # Create metadata
            resume_id = file_path.stem.replace('resume_', '')
            metadata = {
                "filename": file_path.name,
                "resume_id": resume_id,
                "text_length": len(resume_text)
            }
            
            # Add to vector store
            vector_store.add_resume(
                candidate_id=resume_id,
                embedding=embedding,
                document=resume_text,
                metadata=metadata
            )
            
            loaded += 1
            if loaded % 10 == 0:
                print(f"Loaded {loaded} resumes...")
                
        except Exception as e:
            print(f"Failed to load {file_path.name}: {str(e)}")
    
    print(f"✅ Successfully loaded {loaded} resumes to vector store!")
    return loaded


# Quick usage functions
def load_from_processed():
    """Load from data/processed_resumes/"""
    return load_resumes_simple("data/processed_resumes")

def load_from_vector():
    """Load from data/vector_resumes/"""
    return load_resumes_simple("data/vector_resumes")

def load_from_txt():
    """Load from data/txt_resumes/"""
    return load_resumes_simple("data/txt_resumes")


if __name__ == "__main__":
    # Try different directories
    for func, name in [
        (load_from_processed, "processed_resumes"),
        (load_from_vector, "vector_resumes"), 
        (load_from_txt, "txt_resumes")
    ]:
        try:
            count = func()
            if count > 0:
                print(f"✅ Loaded {count} resumes from {name}")
                break
        except Exception as e:
            print(f"❌ Failed to load from {name}: {str(e)}")
    else:
        print("No resume files found in any directory")
