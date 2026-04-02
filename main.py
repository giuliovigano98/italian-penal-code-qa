#!/usr/bin/env python3
"""
Italian Penal Code Q&A System with Ollama LLM
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import click
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from pdf_processor import PDFProcessor
from ollama_client import OllamaClient
from qa_system import QASystem

# Load environment variables
load_dotenv()

console = Console()

@click.group()
def cli():
    """Italian Penal Code Q&A System using Ollama LLM"""
    pass

@cli.command()
@click.option('--pdf-path', default=None, help='Path to the PDF file')
@click.option('--use-transformer', is_flag=True, help='Use transformer training')
def setup(pdf_path, use_transformer):
    """Setup and process the PDF document"""
    console.print(Panel.fit("📚 Setting up Penal Code Q&A System", style="bold blue"))
    
    pdf_path = pdf_path or os.getenv('PDF_PATH')
    if not pdf_path or not os.path.exists(pdf_path):
        console.print("❌ PDF file not found. Please check the path.", style="red")
        return
    
    try:
        # Process PDF
        processor = PDFProcessor()
        console.print("🔄 Processing PDF...")
        chunks = processor.process_pdf(pdf_path)
        console.print(f"✅ Processed {len(chunks)} text chunks", style="green")
        
        if use_transformer:
            # Use transformer training
            console.print("🤖 Training transformer model...", style="yellow")
            from transformer_trainer import LegalEmbeddingSystem
            
            embedding_system = LegalEmbeddingSystem()
            console.print("🔄 This may take 10-20 minutes...")
            embedding_system.train_and_build_embeddings(chunks)
            console.print("✅ Transformer model trained successfully", style="green")
        else:
            # Use traditional TF-IDF
            qa_system = QASystem()
            console.print("🔄 Building vector database...")
            qa_system.build_vector_store(chunks)
            console.print("✅ Vector database built successfully", style="green")
        
        console.print("🎉 Setup complete! You can now ask questions.", style="bold green")
        
    except Exception as e:
        console.print(f"❌ Error during setup: {str(e)}", style="red")

@cli.command()
@click.argument('question')
@click.option('--use-transformer', is_flag=True, help='Use transformer embeddings')
def ask(question, use_transformer):
    """Ask a question about the Italian Penal Code"""
    console.print(Panel.fit(f"🤔 Question: {question}", style="bold yellow"))
    
    try:
        if use_transformer:
            # Use transformer-based system
            from transformer_trainer import LegalEmbeddingSystem
            
            embedding_system = LegalEmbeddingSystem()
            if not embedding_system.load_embeddings():
                console.print("❌ Transformer system not ready. Please run 'setup --use-transformer' first.", style="red")
                return
            
            console.print("🔄 Processing question...")
            relevant_chunks = embedding_system.search_similar(question, top_k=3)
        else:
            # Use traditional system
            qa_system = QASystem()
            if not qa_system.is_ready():
                console.print("❌ System not ready. Please run 'setup' first.", style="red")
                return
            
            console.print("🔄 Processing question...")
            relevant_chunks = qa_system.search_relevant_chunks(question, top_k=3)
        
        if not relevant_chunks:
            console.print("❌ No relevant information found.", style="red")
            return
        
        # Generate answer using Ollama
        ollama_client = OllamaClient()
        context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
        answer = ollama_client.generate_response(question, context)
        
        console.print(Panel.fit(
            Text(answer, style="white"),
            title="💡 Answer",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error: {str(e)}", style="red")

@cli.command()
@click.option('--use-transformer', is_flag=True, help='Use transformer embeddings')
def interactive(use_transformer):
    """Start interactive Q&A session"""
    system_type = "Transformer-based" if use_transformer else "TF-IDF-based"
    console.print(Panel.fit(f"🚀 Interactive Penal Code Q&A ({system_type})", style="bold cyan"))
    console.print("Type 'quit' or 'exit' to end the session.\n")
    
    try:
        if use_transformer:
            from transformer_trainer import LegalEmbeddingSystem
            qa_system = LegalEmbeddingSystem()
            if not qa_system.load_embeddings():
                console.print("❌ Transformer system not ready. Please run 'setup --use-transformer' first.", style="red")
                return
        else:
            qa_system = QASystem()
            if not qa_system.is_ready():
                console.print("❌ System not ready. Please run 'setup' first.", style="red")
                return
        
        ollama_client = OllamaClient()
        
        while True:
            try:
                question = console.input("[bold blue]Ask a question:[/bold blue] ")
                
                if question.lower() in ['quit', 'exit', 'q']:
                    console.print("👋 Goodbye!", style="green")
                    break
                
                if not question.strip():
                    continue
                
                console.print("🔄 Processing...")
                
                # Search for relevant chunks
                if use_transformer:
                    relevant_chunks = qa_system.search_similar(question, top_k=3)
                else:
                    relevant_chunks = qa_system.search_relevant_chunks(question, top_k=3)
                
                if not relevant_chunks:
                    console.print("❌ No relevant information found.", style="red")
                    continue
                
                # Generate answer
                context = "\n\n".join([chunk['text'] for chunk in relevant_chunks])
                answer = ollama_client.generate_response(question, context)
                
                console.print(Panel.fit(
                    Text(answer, style="white"),
                    title="💡 Answer",
                    border_style="green"
                ))
                
            except KeyboardInterrupt:
                console.print("\n👋 Goodbye!", style="green")
                break
            except Exception as e:
                console.print(f"❌ Error: {str(e)}", style="red")
                
    except Exception as e:
        console.print(f"❌ Error initializing system: {str(e)}", style="red")

if __name__ == "__main__":
    cli()
