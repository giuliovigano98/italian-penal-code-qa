"""
PDF Processing Module for Italian Penal Code
"""

import PyPDF2
from typing import List, Dict, Any
import re
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """Process PDF documents and extract text chunks"""
    
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process PDF file and return text chunks with metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Extract text from PDF
            full_text = self._extract_text_from_pdf(pdf_path)
            
            # Clean and structure text
            cleaned_text = self._clean_text(full_text)
            
            # Split into chunks
            chunks = self._split_into_chunks(cleaned_text)
            
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract raw text from PDF file"""
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            logger.info(f"PDF has {num_pages} pages")
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                page_text = page.extract_text()
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text + "\n"
        
        return text
    
    def _clean_text(self, text: str) -> str:
        """Clean and structure extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Add space between lowercase and uppercase
        text = re.sub(r'(\d)([A-Z])', r'\1 \2', text)  # Space between number and letter
        text = re.sub(r'([A-Z])(\d)', r'\1 \2', text)  # Space between letter and number
        
        # Fix article numbering patterns (common in legal documents)
        text = re.sub(r'Art\.(\s*)(\d+)', r'Art. \2', text)
        text = re.sub(r'(\d+)[°º]', r'\1°', text)  # Normalize degree symbols
        
        # Remove page separators
        text = re.sub(r'--- Page \d+ ---', '\n', text)
        
        return text.strip()
    
    def _split_into_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Split text into overlapping chunks with metadata"""
        chunks = []
        
        # Split by articles first (legal documents are usually structured by articles)
        article_pattern = r'(Art\.\s*\d+[a-z°º]*(?:[^.]|\.(?!\d))*?)(?=Art\.|$)'
        articles = re.findall(article_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if not articles:
            # Fallback: split by paragraphs
            paragraphs = text.split('\n\n')
            articles = [p.strip() for p in paragraphs if p.strip()]
        
        current_chunk = ""
        chunk_id = 0
        
        for article in articles:
            article = article.strip()
            if not article:
                continue
            
            # If adding this article would exceed chunk size, save current chunk
            if len(current_chunk) + len(article) > self.chunk_size and current_chunk:
                chunk_data = self._create_chunk(current_chunk, chunk_id)
                chunks.append(chunk_data)
                chunk_id += 1
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-self.overlap:] if len(words) > self.overlap else words
                current_chunk = " ".join(overlap_words) + "\n\n" + article
            else:
                if current_chunk:
                    current_chunk += "\n\n" + article
                else:
                    current_chunk = article
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_data = self._create_chunk(current_chunk, chunk_id)
            chunks.append(chunk_data)
        
        return chunks
    
    def _create_chunk(self, text: str, chunk_id: int) -> Dict[str, Any]:
        """Create chunk dictionary with metadata"""
        # Extract article numbers for better context
        articles = re.findall(r'Art\.\s*(\d+[a-z°º]*)', text, re.IGNORECASE)
        
        return {
            'id': chunk_id,
            'text': text,
            'articles': articles,
            'word_count': len(text.split()),
            'char_count': len(text)
        }
    
    def get_document_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the processed document"""
        total_words = sum(chunk['word_count'] for chunk in chunks)
        total_chars = sum(chunk['char_count'] for chunk in chunks)
        all_articles = set()
        
        for chunk in chunks:
            all_articles.update(chunk['articles'])
        
        return {
            'total_chunks': len(chunks),
            'total_words': total_words,
            'total_characters': total_chars,
            'unique_articles': len(all_articles),
            'articles': sorted(list(all_articles))
        }
