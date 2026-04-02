"""
Q&A System for Italian Penal Code using Vector Search and Ollama
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QASystem:
    """Question Answering system with vector search"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.vector_store_path = self.data_dir / "vector_store.pkl"
        self.chunks_path = self.data_dir / "chunks.pkl"
        
        self.ollama_client = OllamaClient()
        self.vectorizer = None
        self.embeddings = None
        self.chunks = None
        
        # Try to load existing data
        self._load_data()
    
    def _load_data(self):
        """Load existing vector store and chunks"""
        try:
            if self.vector_store_path.exists() and self.chunks_path.exists():
                with open(self.vector_store_path, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data['embeddings']
                    self.vectorizer = data['vectorizer']
                with open(self.chunks_path, 'rb') as f:
                    self.chunks = pickle.load(f)
                
                logger.info(f"Loaded {len(self.chunks)} chunks from disk")
                return True
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
        
        return False
    
    def _save_data(self):
        """Save vector store and chunks to disk"""
        try:
            data_to_save = {
                'embeddings': self.embeddings,
                'vectorizer': self.vectorizer
            }
            with open(self.vector_store_path, 'wb') as f:
                pickle.dump(data_to_save, f)
            with open(self.chunks_path, 'wb') as f:
                pickle.dump(self.chunks, f)
            
            logger.info("Data saved to disk")
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
    
    def build_vector_store(self, chunks: List[Dict[str, Any]]):
        """Build vector store from text chunks"""
        logger.info("Building vector store...")
        
        self.chunks = chunks
        
        # Extract text from chunks
        texts = [chunk['text'] for chunk in chunks]
        
        # Use TF-IDF for vectorization (more reliable than embeddings for this use case)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words=None,  # Keep Italian legal terms
            ngram_range=(1, 2),  # Use bigrams for better context
            lowercase=True
        )
        
        self.embeddings = self.vectorizer.fit_transform(texts)
        
        # Save to disk
        self._save_data()
        
        logger.info(f"Vector store built with {self.embeddings.shape} dimensions")
    
    def search_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for most relevant chunks using hybrid approach (TF-IDF + keyword)"""
        if self.vectorizer is None or self.embeddings is None:
            raise ValueError("Vector store not built. Call build_vector_store first.")
        
        # First try TF-IDF search
        try:
            query_embedding = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
            
            # Get TF-IDF results
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            relevant_chunks = []
            
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Minimum similarity threshold
                    chunk = self.chunks[idx].copy()
                    chunk['similarity_score'] = similarities[idx]
                    relevant_chunks.append(chunk)
        except:
            relevant_chunks = []
        
        # If TF-IDF doesn't find good results, try keyword search
        if len(relevant_chunks) == 0:
            relevant_chunks = self._keyword_search(query, top_k)
        
        return relevant_chunks
    
    def _keyword_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback keyword search"""
        query_words = set(query.lower().split())
        scored_chunks = []
        
        for chunk in self.chunks:
            chunk_text = chunk['text'].lower()
            score = 0
            
            # Count matching words
            for word in query_words:
                if word in chunk_text:
                    score += chunk_text.count(word)
            
            if score > 0:
                chunk_copy = chunk.copy()
                chunk_copy['similarity_score'] = score / len(chunk_text.split())
                scored_chunks.append(chunk_copy)
        
        # Sort by score and return top_k
        scored_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_chunks[:top_k]
    
    def ask_question(self, question: str, top_k: int = 3) -> str:
        """Ask a question and get an answer"""
        try:
            # Search for relevant chunks
            relevant_chunks = self.search_relevant_chunks(question, top_k)
            
            if not relevant_chunks:
                return "Non ho trovato informazioni rilevanti nel Codice Penale per rispondere alla tua domanda."
            
            # Build context from relevant chunks
            context = self._build_context(relevant_chunks)
            
            # Generate answer using Ollama
            answer = self.ollama_client.generate_response(question, context)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error asking question: {str(e)}")
            return f"Mi dispiace, si è verificato un errore: {str(e)}"
    
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from relevant chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"--- Contesto {i} (Punteggio: {chunk['similarity_score']:.3f}) ---")
            context_parts.append(chunk['text'])
            
            # Add article information if available
            if chunk.get('articles'):
                context_parts.append(f"Articoli rilevanti: {', '.join(chunk['articles'])}")
            
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def is_ready(self) -> bool:
        """Check if the system is ready for questions"""
        return self.chunks is not None and self.embeddings is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        if not self.is_ready():
            return {"status": "not_ready"}
        
        return {
            "status": "ready",
            "total_chunks": len(self.chunks),
            "vector_dimensions": self.embeddings.shape,
            "total_articles": len(set(article for chunk in self.chunks for article in chunk.get('articles', [])))
        }
    
    def test_ollama_connection(self) -> bool:
        """Test connection to Ollama"""
        return self.ollama_client.test_connection()
    
    def setup_ollama(self) -> bool:
        """Setup Ollama by pulling required models"""
        if not self.test_ollama_connection():
            logger.warning("Cannot connect to Ollama. Make sure Ollama is running.")
            return False
        
        # Try to pull the model if needed
        if not self.ollama_client.test_connection():
            return self.ollama_client.pull_model()
        
        return True
