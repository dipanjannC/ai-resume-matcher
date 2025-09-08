from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationException
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.batch_size = 32  # Default batch size for batch processing
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info("Loading embedding model", model_name=self.model_name)
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Successfully loaded embedding model")
        except Exception as e:
            logger.error("Failed to load embedding model", error=str(e))
            raise EmbeddingGenerationException(f"Failed to load model {self.model_name}: {str(e)}")
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                raise EmbeddingGenerationException("Empty text provided")
            
            # Clean and normalize text
            text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list of floats
            if isinstance(embedding, np.ndarray):
                embedding_list = embedding.tolist()
            else:
                # Handle tensor case or other types by converting to numpy first
                embedding_list = np.array(embedding).tolist()
            
            logger.debug("Generated embedding", text_length=len(text), embedding_dim=len(embedding_list))
            return embedding_list
            
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise EmbeddingGenerationException(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""
        try:
            if not texts:
                return []
            
            # Clean and normalize texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings
            embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
            
            # Convert to list of lists
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            logger.info("Generated batch embeddings", batch_size=len(texts), 
                       embedding_dim=len(embeddings[0]) if embeddings else 0)
            
            return embeddings
            
        except Exception as e:
            logger.error("Failed to generate batch embeddings", error=str(e))
            raise EmbeddingGenerationException(f"Failed to generate batch embeddings: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation"""
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Truncate if too long (sentence-transformers typically have limits)
        max_length = 512  # Common limit for many models
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        return text
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            # Convert to numpy arrays
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            
            # Ensure similarity is between -1 and 1
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error("Failed to calculate similarity", error=str(e))
            raise EmbeddingGenerationException(f"Failed to calculate similarity: {str(e)}")


# Global embedding service instance
embedding_service = EmbeddingService()
