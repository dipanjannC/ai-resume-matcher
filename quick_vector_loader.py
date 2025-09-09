"""
Minimal Vector Loader - Copy & Run
Ultra-simple script to load resume .txt files into vector store
"""

# Quick setup
import sys; sys.path.append('.')
from pathlib import Path
from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService

# Initialize
vs = VectorStore()
es = EmbeddingService()

# Configuration
RESUME_DIR = "data/processed_resumes"  # Change this path as needed

# Load function
def quick_load():
    files = list(Path(RESUME_DIR).glob("*.txt"))
    print(f"Loading {len(files)} files...")
    
    for i, f in enumerate(files):
        text = f.read_text(encoding='utf-8')
        embedding = es.generate_embedding(text)
        resume_id = f.stem.replace('resume_', '')
        
        vs.add_resume(
            candidate_id=resume_id,
            embedding=embedding, 
            document=text,
            metadata={"filename": f.name, "resume_id": resume_id}
        )
        
        if (i+1) % 10 == 0:
            print(f"Loaded {i+1}/{len(files)}")
    
    print(f"✅ Done! Loaded {len(files)} resumes")

# Run it
if __name__ == "__main__":
    quick_load()
