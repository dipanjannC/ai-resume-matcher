#!/usr/bin/env python3
"""
Resume Vector Loader
Loads processed resume .txt files into ChromaDB vector store for similarity search.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import re

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from app.core.logging import get_logger

logger = get_logger(__name__)


class ResumeVectorLoader:
    """Loads resume .txt files into vector database"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        
    def extract_metadata(self, resume_text: str, filename: str) -> Dict[str, Any]:
        """Extract metadata from resume text"""
        metadata = {
            "filename": filename,
            "resume_id": self._extract_resume_id(filename),
            "text_length": len(resume_text),
        }
        
        # Extract key sections for metadata
        if "SKILLS:" in resume_text:
            skills_section = self._extract_section(resume_text, "SKILLS:")
            metadata["has_skills"] = bool(skills_section.strip())
            
        if "EDUCATION:" in resume_text:
            education_section = self._extract_section(resume_text, "EDUCATION:")
            metadata["has_education"] = bool(education_section.strip())
            
        if "EXPERIENCE:" in resume_text:
            experience_section = self._extract_section(resume_text, "EXPERIENCE:")
            metadata["has_experience"] = bool(experience_section.strip())
            
        # Extract degree level if available
        if "degree" in resume_text.lower():
            if "phd" in resume_text.lower() or "ph.d" in resume_text.lower():
                metadata["education_level"] = "PhD"
            elif "master" in resume_text.lower() or "m.s" in resume_text.lower() or "mba" in resume_text.lower():
                metadata["education_level"] = "Masters"
            elif "bachelor" in resume_text.lower() or "b.s" in resume_text.lower() or "b.tech" in resume_text.lower():
                metadata["education_level"] = "Bachelors"
            else:
                metadata["education_level"] = "Other"
        
        return metadata
    
    def _extract_resume_id(self, filename: str) -> str:
        """Extract resume ID from filename"""
        match = re.search(r'resume_(\d+)', filename)
        return match.group(1) if match else filename.replace('.txt', '')
    
    def _extract_section(self, text: str, section_header: str) -> str:
        """Extract text from a specific section"""
        try:
            start_idx = text.find(section_header)
            if start_idx == -1:
                return ""
            
            # Find the next section header or end of text
            next_sections = ["OBJECTIVE:", "SKILLS:", "EDUCATION:", "EXPERIENCE:", "RESPONSIBILITIES:", "LANGUAGES:", "CERTIFICATIONS:"]
            section_text = text[start_idx + len(section_header):]
            
            for next_section in next_sections:
                if next_section != section_header and next_section in section_text:
                    end_idx = section_text.find(next_section)
                    section_text = section_text[:end_idx]
                    break
            
            return section_text.strip()
        except Exception:
            return ""
    
    def load_resume_file(self, file_path: Path) -> bool:
        """Load a single resume file into vector store"""
        try:
            # Read resume text
            with open(file_path, 'r', encoding='utf-8') as f:
                resume_text = f.read().strip()
            
            if not resume_text:
                logger.warning(f"Empty resume file: {file_path}")
                return False
            
            # Generate embedding
            logger.debug(f"Generating embedding for {file_path.name}")
            embedding = self.embedding_service.generate_embedding(resume_text)
            
            # Extract metadata
            metadata = self.extract_metadata(resume_text, file_path.name)
            
            # Add to vector store
            candidate_id = metadata["resume_id"]
            self.vector_store.add_resume(
                candidate_id=candidate_id,
                embedding=embedding,
                document=resume_text,
                metadata=metadata
            )
            
            logger.info(f"Successfully loaded resume: {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load resume {file_path.name}: {str(e)}")
            return False
    
    def load_resume_directory(self, resume_dir: str, file_pattern: str = "*.txt") -> Dict[str, Any]:
        """Load all resume files from a directory"""
        resume_path = Path(resume_dir)
        
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
        
        # Find all resume files
        resume_files = list(resume_path.glob(file_pattern))
        
        if not resume_files:
            logger.warning(f"No resume files found in {resume_dir} with pattern {file_pattern}")
            return {"loaded": 0, "failed": 0, "total": 0}
        
        logger.info(f"Found {len(resume_files)} resume files to process")
        
        # Process files
        loaded_count = 0
        failed_count = 0
        
        for i, file_path in enumerate(resume_files):
            logger.info(f"Processing {i+1}/{len(resume_files)}: {file_path.name}")
            
            if self.load_resume_file(file_path):
                loaded_count += 1
            else:
                failed_count += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i+1}/{len(resume_files)} files processed")
        
        results = {
            "loaded": loaded_count,
            "failed": failed_count,
            "total": len(resume_files)
        }
        
        logger.info(f"Loading completed: {loaded_count} loaded, {failed_count} failed, {len(resume_files)} total")
        return results


def load_resumes_to_vector_store(resume_directory: str = "data/processed_resumes"):
    """
    Main function to load resume files into vector store
    
    Args:
        resume_directory: Directory containing .txt resume files
    """
    
    print("🔄 Loading Resumes to Vector Store")
    print("=" * 50)
    
    try:
        # Initialize loader
        loader = ResumeVectorLoader()
        
        # Load resumes
        results = loader.load_resume_directory(resume_directory)
        
        # Display results
        print(f"\n📊 Loading Results:")
        print(f"✅ Successfully loaded: {results['loaded']} resumes")
        print(f"❌ Failed to load: {results['failed']} resumes")
        print(f"📁 Total files processed: {results['total']} files")
        print(f"📍 Source directory: {resume_directory}")
        
        if results['loaded'] > 0:
            print(f"\n🎉 Vector store is ready for similarity search!")
            print(f"💡 You can now run job matching queries against {results['loaded']} resumes")
        
    except Exception as e:
        print(f"❌ Error loading resumes: {str(e)}")
        logger.error(f"Failed to load resumes: {str(e)}")


def main():
    """Run the resume loading process"""
    
    # Configuration - modify as needed
    RESUME_DIRECTORIES = [
        "data/processed_resumes",     # From csv_to_txt_converter.py
        # "data/vector_resumes",        # From simple_csv_converter.py
        # "data/txt_resumes",           # From quick_converter.py
        # "data/resumes"                # Original sample resumes
    ]
    
    print("📄 Resume Vector Store Loader")
    print("=" * 40)
    
    # Find which directory exists and has files
    for resume_dir in RESUME_DIRECTORIES:
        resume_path = Path(resume_dir)
        if resume_path.exists():
            txt_files = list(resume_path.glob("*.txt"))
            if txt_files:
                print(f"📁 Found {len(txt_files)} files in {resume_dir}")
                print(f"🔄 Loading from: {resume_dir}")
                load_resumes_to_vector_store(resume_dir)
                return
    
    print("❌ No resume .txt files found in any of the expected directories:")
    for directory in RESUME_DIRECTORIES:
        print(f"   - {directory}")
    print("\n💡 Run one of the CSV converter scripts first to create .txt files")


if __name__ == "__main__":
    main()
