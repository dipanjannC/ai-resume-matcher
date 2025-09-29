import numpy as np
from typing import List, Optional, Any
from app.core.config import settings
from app.core.exceptions import EmbeddingGenerationException
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.device = device or settings.EMBEDDING_DEVICE
        self.batch_size = 32
        self.model: Any = None
        self.use_fallback = False
        self.tfidf_vectorizer = None
        self.svd = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model with fallback handling"""
        try:
            logger.info("Loading embedding model", model_name=self.model_name)
            
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name, device=self.device)
                logger.info("Successfully loaded sentence-transformers model")
                self.use_fallback = False
            except Exception as torch_error:
                if "torch.classes" in str(torch_error) or "__path__._path" in str(torch_error):
                    logger.warning(f"PyTorch/Torch classes error detected: {torch_error}")
                    logger.info("Falling back to simple TF-IDF embeddings")
                    self._setup_fallback_embeddings()
                else:
                    raise torch_error
                    
        except Exception as e:
            logger.error("Failed to load any embedding model", error=str(e))
            logger.info("Setting up fallback TF-IDF embeddings")
            self._setup_fallback_embeddings()
    
    def _setup_fallback_embeddings(self):
        """Setup simple TF-IDF based embeddings as fallback"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            self.svd = TruncatedSVD(n_components=384)
            self.use_fallback = True
            self.model = "fallback_tfidf"
            logger.info("Successfully setup fallback TF-IDF embeddings")
            
        except ImportError:
            logger.error("scikit-learn not available for fallback embeddings")
            raise EmbeddingGenerationException("No embedding method available")
    
    def _generate_tfidf_embedding(self, text: str) -> List[float]:
        """Generate TF-IDF based embedding"""
        try:
            if not self.tfidf_vectorizer or not self.svd:
                raise EmbeddingGenerationException("TF-IDF components not initialized")
            
            # If this is the first text, fit the vectorizer
            if not hasattr(self.tfidf_vectorizer, 'vocabulary_'):
                sample_corpus = [
                    text,
                    "software engineer python java developer",
                    "data scientist machine learning python",
                    "project manager agile scrum leadership"
                ]
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(sample_corpus)
                self.svd.fit(tfidf_matrix)
            
            text_vector = self.tfidf_vectorizer.transform([text])
            embedding = self.svd.transform(text_vector)[0]
            
            # Ensure all elements are floats
            return [float(x) for x in embedding]
            
        except Exception as e:
            logger.error(f"Failed to generate TF-IDF embedding: {e}")
            return [float(x) for x in np.random.normal(0, 1, 384)]
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            if not text or not text.strip():
                raise EmbeddingGenerationException("Empty text provided")
            
            text = self._preprocess_text(text)
            
            if self.use_fallback:
                embedding_result = self._generate_tfidf_embedding(text)
            else:
                if self.model is None:
                    raise EmbeddingGenerationException("Model not loaded")
                
                embedding = self.model.encode(text)
                
                if isinstance(embedding, np.ndarray):
                    embedding_result = list(embedding)
                else:
                    embedding_result = list(np.array(embedding).flatten())
            
            # Ensure all elements are floats and log debug info
            logger.debug(f"Generated embedding: type={type(embedding_result)}, length={len(embedding_result)}")
            if len(embedding_result) > 0:
                logger.debug(f"First element type: {type(embedding_result[0])}")
            
            # Convert all elements to float to ensure compatibility
            embedding_result = [float(x) for x in embedding_result]
            return embedding_result
                
        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise EmbeddingGenerationException(f"Failed to generate embedding: {str(e)}")
    
    def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts in batch"""
        try:
            if not texts:
                return []
            
            processed_texts = [self._preprocess_text(text) for text in texts if text.strip()]
            if not processed_texts:
                return []
            
            if self.use_fallback:
                return [self._generate_tfidf_embedding(text) for text in processed_texts]
            else:
                if self.model is None:
                    raise EmbeddingGenerationException("Model not loaded")
                
                embeddings = self.model.encode(processed_texts)
                
                if isinstance(embeddings, np.ndarray):
                    return [[float(x) for x in emb] for emb in embeddings]
                else:
                    return [[float(x) for x in np.array(emb).flatten()] for emb in embeddings]
                
        except Exception as e:
            logger.error("Failed to generate batch embeddings", error=str(e))
            raise EmbeddingGenerationException(f"Failed to generate batch embeddings: {str(e)}")
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text before embedding generation"""
        text = text.strip()
        text = ' '.join(text.split())
        
        max_length = 512
        if len(text.split()) > max_length:
            text = ' '.join(text.split()[:max_length])
        
        return text
    
    def get_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            emb1 = np.array(embedding1)
            emb2 = np.array(embedding2)
            
            dot_product = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            similarity = np.clip(similarity, -1.0, 1.0)
            
            return float(similarity)
            
        except Exception as e:
            logger.error("Failed to calculate similarity", error=str(e))
            return 0.0


# Create singleton instance
embedding_service = EmbeddingService()