"""
Ollama LLM Client for Italian Penal Code Q&A
"""

import os
import ollama
from typing import List, Dict, Any, Optional
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama LLM"""
    
    def __init__(self):
        self.base_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
        self.model = os.getenv('OLLAMA_MODEL', 'llama3.2')
        
        # Configure ollama client
        ollama.host = self.base_url
        
        logger.info(f"Initialized Ollama client with model: {self.model}")
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model in available_models:
                logger.info(f"Model {self.model} is available")
                return True
            else:
                logger.warning(f"Model {self.model} not found. Available models: {available_models}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            return False
    
    def pull_model(self) -> bool:
        """Pull the specified model if not available"""
        try:
            logger.info(f"Pulling model {self.model}...")
            ollama.pull(self.model)
            logger.info("Model pulled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to pull model: {str(e)}")
            return False
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """
        Generate response from Ollama LLM
        
        Args:
            prompt: The user's question
            context: Optional context from retrieved documents
            
        Returns:
            Generated response
        """
        try:
            # Build the full prompt with context if provided
            full_prompt = self._build_prompt(prompt, context)
            
            # Generate response
            response = ollama.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': 0.1,  # Lower temperature for more factual responses
                    'top_p': 0.9,
                    'max_tokens': 2000
                }
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _build_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build the complete prompt for the LLM"""
        
        system_prompt = """Sei un assistente esperto del Codice Penale Italiano. 
Il tuo compito è rispondere alle domande basandoti esclusivamente sul contesto fornito.
Se il contesto non contiene informazioni sufficienti per rispondere, indica chiaramente 
che non hai abbastanza informazioni.

Regole:
1. Rispondi solo in italiano
2. Basa le risposte esclusivamente sul contesto fornito
3. Se non trovi la risposta nel contesto, dillo chiaramente
4. Sii preciso e cita gli articoli rilevanti quando possibile
5. Non inventare informazioni non presenti nel testo
"""
        
        if context:
            full_prompt = f"""{system_prompt}

CONTESTO DEL CODICE PENALE:
{context}

DOMANDA: {question}

RISPOSTA:"""
        else:
            full_prompt = f"""{system_prompt}

DOMANDA: {question}

RISPOSTA:"""
        
        return full_prompt
    
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Ollama
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        try:
            # Use a model that supports embeddings
            embedding_model = os.getenv('EMBEDDING_MODEL', 'llama3.2')
            
            response = ollama.embeddings(
                model=embedding_model,
                prompt=text
            )
            
            return response['embedding']
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Fallback: return a simple hash-based embedding
            import hashlib
            hash_obj = hashlib.md5(text.encode())
            # Convert hash to float values
            hash_hex = hash_obj.hexdigest()
            embedding = []
            for i in range(0, len(hash_hex), 2):
                byte_val = int(hash_hex[i:i+2], 16)
                embedding.append(byte_val / 255.0)
            
            # Pad or truncate to standard size (768 dimensions)
            while len(embedding) < 768:
                embedding.append(0.0)
            return embedding[:768]
    
    def chat_completion(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response using chat completion format
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Generated response
        """
        try:
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    'temperature': 0.1,
                    'top_p': 0.9,
                    'max_tokens': 2000
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise
