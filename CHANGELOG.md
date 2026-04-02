# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-04-02

### Added
- Initial release of Italian Penal Code Q&A System
- PDF processing with intelligent chunking
- TF-IDF based vector search system
- Ollama LLM integration
- Transformer training for legal domain
- CLI interface with rich terminal UI
- Interactive Q&A mode
- Hybrid search (TF-IDF + keyword fallback)
- Legal text cleaning and preprocessing
- Special tokens for legal domain ([ART], [PENA], [DELITTO], [REATO])
- Support for Italian legal terminology
- Configuration via environment variables
- Comprehensive documentation

### Features
- **PDF Processing**: Extract and chunk legal documents by articles
- **Dual Search Modes**: TF-IDF and transformer-based embeddings
- **LLM Integration**: Local Ollama model for answer generation
- **CLI Tools**: Setup, ask questions, interactive mode
- **Model Training**: Fine-tune BERT on Italian legal text
- **Semantic Search**: Advanced embeddings for legal concepts

### Technical
- Python 3.8+ support
- Virtual environment setup
- Modular architecture
- Error handling and logging
- Performance optimization
- Memory management

### Documentation
- Complete README with setup instructions
- API documentation
- Contributing guidelines
- License and disclaimer

---

## [Unreleased]

### Planned
- Web interface
- API endpoints
- Multi-language support
- Additional legal documents
- Performance improvements
- More LLM providers
