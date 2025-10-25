# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-25

### Added

#### Core Features
- RAG Pipeline implementation with MongoDB Atlas Vector Search
- LangChain integration for document processing and retrieval
- Context manager support for automatic resource cleanup
- Configurable retrieval and generation pipeline

#### Components
- **EmbeddingsManager**: Manages embedding model initialization
  - OpenAI embeddings support (text-embedding-3-small, text-embedding-3-large)
  - Extensible architecture for additional providers
- **LLMManager**: Manages language model initialization
  - OpenAI LLM support (gpt-4o-mini, gpt-4, etc.)
  - Configurable temperature and token limits
- **VectorStoreManager**: MongoDB Atlas vector store management
  - Document addition and retrieval
  - Similarity search with optional score thresholds
  - Connection pooling and reuse
- **DocumentProcessor**: Document chunking and processing
  - Recursive character text splitting
  - Configurable chunk size and overlap
  - Support for multiple file formats

#### Configuration
- Pydantic V2 models for type-safe configuration
  - RAGPipelineConfig: Main pipeline configuration
  - MongoDBConfig: MongoDB connection and database settings
  - EmbeddingConfig: Embedding model configuration
  - LLMConfig: LLM configuration
  - RetrieverConfig: Retrieval parameters
  - ChunkingConfig: Document chunking settings
- YAML-based configuration files
- Environment variable substitution with defaults
- Configuration validation with detailed error messages

#### Utilities
- Configuration loader with YAML parsing
- Deep merge for configuration overrides
- Environment variable substitution
- Logger setup with file and console handlers
- Document processing utilities

#### Testing
- Unit tests for configuration models
- Unit tests for configuration loader
- Unit tests for document processor
- Pytest configuration and setup
- Test fixtures for temporary files

#### Documentation
- Comprehensive README with installation and usage
- Quick Start Guide (QUICKSTART.md)
- Implementation details (IMPLEMENTATION.md)
- Example scripts:
  - Basic usage example
  - Document processing example
  - Custom configuration example
- Inline docstrings for all functions and classes
- Type hints throughout the codebase

#### Code Quality
- PEP-8 compliant code style
- Black code formatting
- Ruff linting with strict rules
- MyPy type checking support
- Full type annotations
- Lazy imports for better test performance
- CodeQL security scanning (0 alerts)

#### Project Structure
- Modern Python project layout
- pyproject.toml for project configuration
- requirements.txt for dependencies
- .gitignore for Python projects
- .env.example for environment variables

#### Dependencies
- langchain>=0.3.0
- langchain-community>=0.3.0
- langchain-mongodb>=0.2.0
- langchain-openai>=0.2.0
- pymongo>=4.8.0
- pydantic>=2.9.0
- pydantic-settings>=2.5.0
- python-dotenv>=1.0.0
- pyyaml>=6.0.1
- tiktoken>=0.7.0

#### Development Tools
- pytest>=8.3.0
- pytest-cov>=5.0.0
- pytest-asyncio>=0.24.0
- black>=24.8.0
- ruff>=0.6.0
- mypy>=1.11.0
- pre-commit>=3.8.0

### Technical Details

#### Architecture Decisions
- Modular component design for easy extension
- Lazy imports to avoid heavy dependency loading
- Context manager pattern for resource management
- Factory pattern for component initialization
- Configuration-driven design
- Type-safe configuration with Pydantic V2

#### Performance Optimizations
- Connection pooling for MongoDB
- Lazy initialization of components
- Efficient document chunking
- Batch document processing support

#### Security Features
- No hardcoded credentials
- Environment variable-based configuration
- Pydantic validation for all inputs
- CodeQL security scanning
- Latest dependency versions

### Python Version Support
- Python 3.12+ required
- Modern Python features utilized
- Type hints using Python 3.12 syntax

### Testing
- 23 unit tests passing
- Configuration validation tests
- Configuration loader tests
- Document processor tests
- Test coverage for core functionality

### Code Quality Metrics
- ✅ Black formatting: 100% compliant
- ✅ Ruff linting: All checks passed
- ✅ Type checking: MyPy compatible
- ✅ Security: 0 CodeQL alerts
- ✅ Tests: 23/23 passing

### Known Limitations
- Currently supports only OpenAI embeddings and LLMs
- MongoDB Atlas is the only supported vector store
- Synchronous operations only (no async support yet)
- English language focus

### Future Plans
See IMPLEMENTATION.md for planned enhancements:
- Additional vector stores (Chroma, Pinecone)
- More embedding providers (HuggingFace, Cohere)
- More LLM providers (Anthropic, local models)
- Tool chaining capabilities
- Agent-based workflows
- Async support
- Multi-modal RAG
- Evaluation metrics

[0.1.0]: https://github.com/vijulshah/AgenticPipelines/releases/tag/v0.1.0
