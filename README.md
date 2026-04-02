# Italian Penal Code Q&A System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai)

A powerful question-answering system that uses LLMs and transformer models to answer questions about the Italian Penal Code based on PDF documents.

## 🚀 Features

- 📚 **PDF Processing**: Intelligent text extraction and chunking from legal documents
- 🤖 **LLM Integration**: Local Ollama integration for natural language understanding
- 🔍 **Dual Search Modes**: TF-IDF and transformer-based semantic search
- 🧠 **Custom Transformer**: BERT model fine-tuned on Italian legal text
- 💬 **Rich CLI Interface**: Beautiful terminal UI with interactive mode
- 🇮🇹 **Italian Optimized**: Specialized for Italian legal terminology
- ⚡ **Hybrid Search**: Combines vector search with keyword fallback
- 🎯 **Legal Domain**: Special tokens for articles, penalties, and crimes

## 📋 Prerequisites

- **Python 3.8+**
- **Ollama** installed and running locally
- **Italian Penal Code PDF** (or any legal PDF document)

### Installing Ollama

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama server
ollama serve

# Pull the recommended model
ollama pull llama3.2
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/italian-penal-code-qa.git
   cd italian-penal-code-qa
   ```

2. **Create virtual environment**
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## 🚀 Quick Start

### 1. Setup the System

Process the PDF and build the vector database:

```bash
# Fast TF-IDF mode
python main.py setup

# Advanced transformer mode (recommended)
python main.py setup --use-transformer
```

### 2. Ask Questions

#### Single Question
```bash
python main.py ask "Qual è la pena per l'omicidio?"
python main.py ask "Cos'è il furto e quali sono le pene?" --use-transformer
```

#### Interactive Mode
```bash
python main.py interactive
python main.py interactive --use-transformer
```

## 📖 Usage Examples

### Example Questions
```bash
python main.py ask "Qual è la pena per il furto?" --use-transformer
python main.py ask "Cos'è l'omicidio colposo?" --use-transformer
python main.py ask "Quali sono gli elementi del reato di truffa?" --use-transformer
python main.py ask "Cosa prevede l'articolo 575 del codice penale?" --use-transformer
python main.py ask "Qual è la differenza tra dolo e colpa?" --use-transformer
```

### Available Commands

```bash
python main.py --help
# Commands:
#   setup          Setup and process the PDF document
#   ask            Ask a question about the Italian Penal Code
#   interactive    Start interactive Q&A session

# Options:
#   --use-transformer    Use transformer-based embeddings (more accurate)
#   --pdf-path PATH      Custom PDF file path
```

## 🏗️ Architecture

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Input     │───▶│  PDF Processor  │───▶│   Text Chunks   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Question  │───▶│  Search System  │◀───│  Embeddings     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Ollama LLM     │
                       └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Answer        │
                       └─────────────────┘
```

### Core Modules

1. **PDF Processor** (`src/pdf_processor.py`)
   - Extracts text from PDF files
   - Intelligent chunking based on legal structure
   - Text cleaning and normalization

2. **Ollama Client** (`src/ollama_client.py`)
   - Interface to Ollama LLM
   - Prompt engineering for legal Q&A
   - Embedding generation

3. **QA System** (`src/qa_system.py`)
   - TF-IDF vector-based semantic search
   - Context retrieval and ranking
   - Answer generation

4. **Transformer Trainer** (`src/transformer_trainer.py`)
   - BERT fine-tuning on legal text
   - Legal domain-specific embeddings
   - Special token handling

## 📊 Performance

### TF-IDF Mode
- **Setup Time**: ~30 seconds
- **Query Time**: ~1-2 seconds
- **Accuracy**: Good for exact matches

### Transformer Mode
- **Setup Time**: ~10-15 minutes (training)
- **Query Time**: ~3-5 seconds
- **Accuracy**: Excellent for semantic understanding

### Model Statistics
- **2079 chunks** processed from 111 pages
- **696 legal terms** identified
- **768 dimensions** embeddings
- **Loss**: 0.356 after 3 epochs

## ⚙️ Configuration

### Environment Variables

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2

# PDF Path
PDF_PATH=/path/to/your/codicepenale.pdf

# Optional: Embedding Model
EMBEDDING_MODEL=llama3.2
```

### Advanced Configuration

You can modify the chunking parameters in `pdf_processor.py`:
- `chunk_size`: Maximum characters per chunk (default: 1000)
- `overlap`: Overlap between chunks (default: 200)

## 🔧 Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/italian-penal-code-qa.git
cd italian-penal-code-qa

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Setup environment
cp .env.example .env
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest --cov=src tests/
```

### Code Style

```bash
# Format code
black src/ main.py

# Check linting
flake8 src/ main.py

# Type checking
mypy src/ main.py
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas for Contribution

- 🐛 **Bug fixes**: PDF processing, error handling
- ✨ **New features**: Web interface, API endpoints
- 📚 **Documentation**: Tutorials, examples
- 🧪 **Testing**: Unit tests, integration tests
- 🌐 **Multi-language**: Support for other languages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. Legal advice should be obtained from qualified legal professionals.

## 🙏 Acknowledgments

- **Ollama** for local LLM infrastructure
- **Hugging Face** for transformer models
- **dbmdz** for Italian BERT models
- **Rich** for beautiful terminal UI

## 📞 Support

- 📧 Email: support@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/italian-penal-code-qa/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/italian-penal-code-qa/discussions)

---

⭐ If you find this project useful, please give it a star on GitHub!
