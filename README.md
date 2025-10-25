# AgenticPipelines

A space for Researching and Developing Agentic Pipelines with Different development Frameworks, logging pipelines, RAG flows and with various models.

## Features

- 🚀 **RAG Pipeline with MongoDB**: Production-ready RAG implementation using LangChain and MongoDB Atlas Vector Search
- 📦 **Pydantic Models**: Type-safe configuration management with validation
- ⚙️ **Dynamic Configuration**: YAML-based config files with environment variable substitution
- 🎯 **PEP-8 Compliant**: Clean, maintainable code following Python best practices
- 🔧 **Modular Design**: Easily extensible architecture for adding new components
- 📝 **Comprehensive Logging**: Built-in logging with file and console handlers
- 🧪 **Type Hints**: Full type annotations for better IDE support and code quality

## Project Structure

```
AgenticPipelines/
├── src/
│   └── rag_pipeline/          # RAG pipeline implementation
│       ├── core/              # Core pipeline components
│       │   ├── embeddings.py  # Embeddings management
│       │   ├── llm.py         # LLM management
│       │   ├── pipeline.py    # Main RAG pipeline
│       │   └── vector_store.py # Vector store management
│       ├── models/            # Pydantic models
│       │   └── config.py      # Configuration models
│       └── utils/             # Utility modules
│           ├── config_loader.py   # Config loading utilities
│           ├── document_processor.py # Document processing
│           └── logger.py      # Logging utilities
├── configs/                   # Configuration files
│   └── default_config.yaml    # Default RAG configuration
├── examples/                  # Example scripts
│   ├── basic_usage.py        # Basic RAG usage
│   ├── document_processing.py # Document loading example
│   └── custom_config.py      # Custom configuration example
├── tests/                     # Test files
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── pyproject.toml           # Project configuration
└── README.md                # This file
```

## Installation

### Prerequisites

- Python 3.12+
- MongoDB (local or MongoDB Atlas)
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/vijulshah/AgenticPipelines.git
cd AgenticPipelines
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys and MongoDB connection string
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from src.rag_pipeline import RAGPipeline, load_config

# Load configuration
config = load_config(Path("configs/default_config.yaml"))

# Initialize pipeline
with RAGPipeline(config) as pipeline:
    # Add documents
    text = "Your document text here..."
    pipeline.add_text(text)
    
    # Query the pipeline
    result = pipeline.query("Your question here?")
    print(result['answer'])
```

### Configuration

Create a YAML configuration file or use the default one in `configs/default_config.yaml`:

```yaml
pipeline_name: "my_rag_pipeline"
vector_store_type: "mongodb"

mongodb:
  connection_string: "${MONGODB_URI}"
  database_name: "rag_database"
  collection_name: "vector_store"
  index_name: "vector_index"

embedding:
  provider: "openai"
  model_name: "text-embedding-3-small"
  dimensions: 1536
  api_key: "${OPENAI_API_KEY}"

llm:
  provider: "openai"
  model_name: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 1000
  api_key: "${OPENAI_API_KEY}"

retriever:
  search_type: "similarity"
  k: 4

chunking:
  chunk_size: 1000
  chunk_overlap: 200
```

### Programmatic Configuration

```python
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.rag_pipeline.models import MongoDBConfig, EmbeddingConfig, LLMConfig

config = RAGPipelineConfig(
    pipeline_name="custom_pipeline",
    mongodb=MongoDBConfig(
        connection_string="mongodb://localhost:27017",
        database_name="my_db",
    ),
    embedding=EmbeddingConfig(
        provider="openai",
        model_name="text-embedding-3-small",
    ),
    llm=LLMConfig(
        provider="openai",
        model_name="gpt-4o-mini",
        temperature=0.5,
    ),
)

pipeline = RAGPipeline(config)
```

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_usage.py`: Simple RAG pipeline setup
- `document_processing.py`: Loading and processing documents
- `custom_config.py`: Using custom configurations

Run an example:
```bash
python examples/basic_usage.py
```

## MongoDB Setup

### Local MongoDB
```bash
# Start MongoDB
mongod --dbpath /path/to/data/directory
```

### MongoDB Atlas

1. Create a free MongoDB Atlas cluster at https://www.mongodb.com/cloud/atlas
2. Create a database user and get the connection string
3. Create a vector search index on your collection:

```json
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 1536,
      "similarity": "cosine"
    }
  ]
}
```

4. Update your `.env` file with the connection string

## Development

### Code Style

This project follows PEP-8 guidelines and uses:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

### Running Linters

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Format code
black src/ examples/ tests/

# Lint code
ruff check src/ examples/ tests/

# Type check
mypy src/
```

## Dependencies

Core dependencies:
- `langchain>=0.3.0` - LLM framework
- `langchain-mongodb>=0.2.0` - MongoDB vector store
- `langchain-openai>=0.2.0` - OpenAI integration
- `pymongo>=4.8.0` - MongoDB driver
- `pydantic>=2.9.0` - Data validation
- `pyyaml>=6.0.1` - YAML parsing

## Roadmap

- [x] RAG Pipeline with MongoDB
- [ ] Tool Chaining Flows
- [ ] Additional Vector Stores (Chroma, Pinecone)
- [ ] Multi-modal RAG
- [ ] Agent-based pipelines
- [ ] Advanced retrieval strategies
- [ ] Evaluation metrics and benchmarking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

Built with:
- [LangChain](https://github.com/langchain-ai/langchain)
- [MongoDB](https://www.mongodb.com/)
- [Pydantic](https://docs.pydantic.dev/)
- [OpenAI](https://openai.com/)
