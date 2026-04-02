"""
Transformer Trainer for Italian Penal Code
"""

import os
import re
import torch
import numpy as np
from typing import List, Dict, Any
from pathlib import Path
import logging
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    TrainingArguments, 
    Trainer,
    BertForMaskedLM,
    BertConfig
)
from torch.utils.data import Dataset
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalTextDataset(Dataset):
    """Dataset for legal text training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class LegalTextCleaner:
    """Clean and preprocess legal text"""
    
    def __init__(self):
        # Italian legal stopwords
        self.legal_stopwords = {
            'codice', 'penale', 'articolo', 'art', 'comma', 'lettera', 'numero',
            'disposizione', 'legge', 'testo', 'precedentemente', 'vigore', 'così',
            'modificato', 'aggiunto', 'sostituito', 'abrogato', 'dal', 'dall',
            'dell', 'dello', 'della', 'dei', 'degli', 'delle', 'del', 'di',
            'a', 'da', 'in', 'con', 'per', 'su', 'tra', 'fra', 'e', 'o', 'ma',
            'se', 'quando', 'come', 'che', 'chi', 'cui', 'quale', 'quanti',
            'questo', 'questa', 'questi', 'queste', 'quello', 'quella', 'quelli',
            'quelle', 'lui', 'lei', 'noi', 'voi', 'loro', 'mio', 'tua', 'suo',
            'nostro', 'vostro', 'loro', 'un', 'una', 'uno', 'il', 'lo', 'la',
            'i', 'gli', 'le', 'è', 'sono', 'stata', 'stati', 'stato', 'state',
            'essere', 'avere', 'fare', 'dire', 'potere', 'dovere', 'volere'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize legal text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep legal punctuation
        text = re.sub(r'[^\w\s\.\,\;\:\-\(\)\[\]]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers
        text = re.sub(r'pagina\s*\d+', '', text)
        text = re.sub(r'codice\s+penale\s+libro\s+[ivx]+', '', text)
        
        # Remove legal stopwords
        words = text.split()
        words = [w for w in words if w not in self.legal_stopwords and len(w) > 2]
        
        return ' '.join(words)
    
    def extract_legal_terms(self, text: str) -> List[str]:
        """Extract important legal terms"""
        # Look for article references
        articles = re.findall(r'art\.?\s*(\d+)', text.lower())
        
        # Look for legal terms
        legal_terms = re.findall(
            r'(reato|pena|reclusione|multa|ammenda|delitto|contravvenzione|'
            r'colpa|dolo|violenza|minaccia|furto|rapina|truffa|appropriazione|'
            r'omicidio|lesioni|calunnia|diffamazione|ingiuria|estorsione|'
            r'corruzione|peculato|abuso|potere|autorità|pubblico|ufficiale)',
            text.lower()
        )
        
        return list(set(articles + legal_terms))

class TransformerTrainer:
    """Train transformer model on legal text"""
    
    def __init__(self, model_name: str = "dbmdz/bert-base-italian-xxl-uncased"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.cleaner = LegalTextCleaner()
        
    def setup_model(self):
        """Setup tokenizer and model"""
        logger.info(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens for legal domain BEFORE creating the model
        special_tokens = ['[ART]', '[PENA]', '[DELITTO]', '[REATO]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        
        # Configure BERT for legal text
        config = BertConfig.from_pretrained(self.model_name)
        
        # Create model with ignore_mismatched_sizes to handle token addition
        self.model = BertForMaskedLM.from_pretrained(
            self.model_name, 
            config=config,
            ignore_mismatched_sizes=True
        )
        
        # Resize embeddings to match new vocabulary size
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        logger.info("Model setup complete")
    
    def prepare_training_data(self, chunks: List[Dict[str, Any]]) -> List[str]:
        """Prepare cleaned training data"""
        logger.info("Preparing training data...")
        
        cleaned_texts = []
        legal_terms_count = {}
        
        for chunk in chunks:
            # Clean the text
            cleaned = self.cleaner.clean_text(chunk['text'])
            
            # Extract legal terms for better training
            terms = self.cleaner.extract_legal_terms(chunk['text'])
            for term in terms:
                legal_terms_count[term] = legal_terms_count.get(term, 0) + 1
            
            # Add legal term markers
            for term in terms:
                if term in ['reato', 'delitto', 'contravvenzione']:
                    cleaned = cleaned.replace(term, '[DELITTO] ' + term)
                elif 'art' in term or term.isdigit():
                    cleaned = cleaned.replace(term, '[ART] ' + term)
                elif 'pena' in term or 'reclusione' in term or 'multa' in term:
                    cleaned = cleaned.replace(term, '[PENA] ' + term)
            
            if len(cleaned.split()) > 10:  # Keep meaningful texts
                cleaned_texts.append(cleaned)
        
        logger.info(f"Prepared {len(cleaned_texts)} training samples")
        logger.info(f"Legal terms found: {len(legal_terms_count)}")
        
        return cleaned_texts
    
    def train_model(self, chunks: List[Dict[str, Any]], output_dir: str = "legal_model"):
        """Train the transformer model"""
        logger.info("Starting model training...")
        
        # Setup model
        self.setup_model()
        
        # Prepare data
        training_texts = self.prepare_training_data(chunks)
        
        # Create dataset
        dataset = LegalTextDataset(training_texts, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="no",
            save_total_limit=2,
            load_best_model_at_end=False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info("Training started...")
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info(f"Model saved to {output_dir}")
        
        return output_dir
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for texts"""
        if not self.model or not self.tokenizer:
            self.setup_model()
        
        # Force CPU to avoid MPS issues on Mac
        device = torch.device('cpu')
        self.model.to(device)
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # Clean text
                cleaned = self.cleaner.clean_text(text)
                
                # Tokenize
                inputs = self.tokenizer(
                    cleaned,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Get embeddings - use the bert base model directly
                bert_outputs = self.model.bert(**inputs)
                hidden_states = bert_outputs.last_hidden_state
                
                # Use last hidden state mean for embeddings
                embedding = hidden_states.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)

class LegalEmbeddingSystem:
    """Enhanced embedding system for legal Q&A"""
    
    def __init__(self, model_path: str = None):
        self.trainer = TransformerTrainer()
        self.embeddings = None
        self.chunks = None
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model()
    
    def load_model(self):
        """Load trained model"""
        if self.model_path:
            self.trainer.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.trainer.model = BertForMaskedLM.from_pretrained(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
    
    def train_and_build_embeddings(self, chunks: List[Dict[str, Any]], output_dir: str = "legal_model"):
        """Train model and build embeddings"""
        # Train the model
        model_path = self.trainer.train_model(chunks, output_dir)
        
        # Load the trained model
        self.load_model()
        
        # Generate embeddings for all chunks
        texts = [chunk['text'] for chunk in chunks]
        self.embeddings = self.trainer.generate_embeddings(texts)
        self.chunks = chunks
        
        # Save embeddings
        self.save_embeddings()
        
        logger.info(f"Built embeddings with shape: {self.embeddings.shape}")
        return self.embeddings
    
    def save_embeddings(self):
        """Save embeddings and chunks"""
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
        
        with open(data_dir / "legal_embeddings.pkl", 'wb') as f:
            pickle.dump(self.embeddings, f)
        
        with open(data_dir / "legal_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        logger.info("Embeddings saved to disk")
    
    def load_embeddings(self):
        """Load embeddings and chunks"""
        data_dir = Path("data")
        
        if (data_dir / "legal_embeddings.pkl").exists():
            with open(data_dir / "legal_embeddings.pkl", 'rb') as f:
                self.embeddings = pickle.load(f)
            
            with open(data_dir / "legal_chunks.pkl", 'rb') as f:
                self.chunks = pickle.load(f)
            
            logger.info("Embeddings loaded from disk")
            return True
        
        return False
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar chunks using cosine similarity"""
        if self.embeddings is None:
            if not self.load_embeddings():
                raise ValueError("No embeddings available. Train model first.")
        
        # Generate query embedding
        query_embedding = self.trainer.generate_embeddings([query])[0]
        
        # Calculate similarities
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx].copy()
            chunk['similarity_score'] = similarities[idx]
            results.append(chunk)
        
        return results
