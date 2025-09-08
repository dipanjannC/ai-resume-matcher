import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional, Any, Union
import uuid
from app.core.config import settings
from app.core.exceptions import VectorStoreException
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self.client = None
        self.collection = None
        self.collection_name = "resume_embeddings"
        self.model_name = "all-MiniLM-L6-v2"
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client"""
        try:
            logger.info("Initializing ChromaDB client", 
                       persist_directory=settings.CHROMADB_PERSIST_DIRECTORY)
            
            self.client = chromadb.PersistentClient(
                path=settings.CHROMADB_PERSIST_DIRECTORY,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Resume embeddings for similarity search"}
            )
            
            logger.info("Successfully initialized ChromaDB", 
                       collection_name=self.collection_name)
            
        except Exception as e:
            logger.error("Failed to initialize ChromaDB", error=str(e))
            raise VectorStoreException(f"Failed to initialize vector store: {str(e)}")
    
    def add_resume(self, candidate_id: str, embedding: List[float], 
                   document: str, metadata: Dict[str, Any]) -> str:
        """Add a resume embedding to the vector store"""
        try:
            if not self.collection:
                raise VectorStoreException("Collection not initialized")
                
            doc_id = str(candidate_id)
            
            self.collection.add(
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.info("Added resume to vector store", candidate_id=doc_id)
            return doc_id
            
        except Exception as e:
            logger.error("Failed to add resume to vector store", 
                        candidate_id=candidate_id, error=str(e))
            raise VectorStoreException(f"Failed to add resume: {str(e)}")
    
    def search_similar(self, query_embedding: List[float], 
                      top_k: int = 10, 
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar resumes"""
        try:
            if not self.collection:
                raise VectorStoreException("Collection not initialized")
                
            where_clause = filter_metadata if filter_metadata else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause,
                include=["metadatas", "documents", "distances"]
            )
            
            formatted_results = []
            if results and results.get('ids') and results['ids'][0]:
                ids = results['ids'][0]
                
                # Safely extract lists with proper None handling
                distances_data = results.get('distances')
                distances = distances_data[0] if distances_data and len(distances_data) > 0 else []
                
                metadatas_data = results.get('metadatas')
                metadatas = metadatas_data[0] if metadatas_data and len(metadatas_data) > 0 else []
                
                documents_data = results.get('documents')
                documents = documents_data[0] if documents_data and len(documents_data) > 0 else []
                
                for i, candidate_id in enumerate(ids):
                    distance = distances[i] if distances and i < len(distances) else 0.0
                    result = {
                        'candidate_id': candidate_id,
                        'distance': distance,
                        'similarity': 1 - distance,
                        'metadata': metadatas[i] if metadatas and i < len(metadatas) else {},
                        'document': documents[i] if documents and i < len(documents) else ""
                    }
                    formatted_results.append(result)
            
            logger.info("Search completed", 
                       query_results=len(formatted_results), top_k=top_k)
            
            return formatted_results
            
        except Exception as e:
            logger.error("Failed to search similar resumes", error=str(e))
            raise VectorStoreException(f"Failed to search: {str(e)}")
    
    def get_all_candidates(self) -> List[Dict[str, Any]]:
        """Get all candidate IDs and metadata"""
        try:
            if not self.collection:
                raise VectorStoreException("Collection not initialized")
            
            results = self.collection.get(include=["metadatas", "documents"])
            
            candidates = []
            if results and results.get('ids'):
                ids = results['ids']
                metadatas_data = results.get('metadatas')
                documents_data = results.get('documents')
                
                for i, candidate_id in enumerate(ids):
                    candidate = {
                        'candidate_id': candidate_id,
                        'metadata': metadatas_data[i] if metadatas_data and i < len(metadatas_data) else {},
                        'document': documents_data[i] if documents_data and i < len(documents_data) else ""
                    }
                    candidates.append(candidate)
            
            logger.info("Retrieved all candidates", count=len(candidates))
            return candidates
            
        except Exception as e:
            logger.error("Failed to get all candidates", error=str(e))
            raise VectorStoreException(f"Failed to get candidates: {str(e)}")
    
    def resume_exists(self, candidate_id: str) -> bool:
        """Check if a resume exists in the vector store"""
        try:
            if not self.collection:
                return False
                
            doc_id = str(candidate_id)
            results = self.collection.get(ids=[doc_id])
            exists = bool(results and results.get('ids') and doc_id in results['ids'])
            
            logger.debug("Checked resume existence", 
                        candidate_id=doc_id, exists=exists)
            return exists
            
        except Exception as e:
            logger.error("Failed to check resume existence", 
                        candidate_id=candidate_id, error=str(e))
            return False
    
    def delete_resume(self, candidate_id: str) -> bool:
        """Delete a resume from the vector store"""
        try:
            if not self.collection:
                return False
                
            doc_id = str(candidate_id)
            self.collection.delete(ids=[doc_id])
            
            logger.info("Deleted resume from vector store", candidate_id=doc_id)
            return True
            
        except Exception as e:
            logger.error("Failed to delete resume", 
                        candidate_id=candidate_id, error=str(e))
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            if not self.collection:
                return {'total_resumes': 0, 'collection_name': self.collection_name}
            
            count = self.collection.count()
            
            stats = {
                'total_resumes': count,
                'collection_name': self.collection_name,
                'model_name': self.model_name
            }
            
            logger.info("Retrieved collection stats", **stats)
            return stats
            
        except Exception as e:
            logger.error("Failed to get collection stats", error=str(e))
            return {'total_resumes': 0, 'collection_name': self.collection_name}


# Create global instance
vector_store = VectorStore()
