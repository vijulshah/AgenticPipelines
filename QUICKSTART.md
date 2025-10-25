# Quick Start Guide

This guide will help you get started with the RAG Pipeline in 5 minutes.

## Prerequisites

- Python 3.12 or higher
- MongoDB (local or Atlas)
- OpenAI API key

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/vijulshah/AgenticPipelines.git
cd AgenticPipelines
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
```

Edit `.env` and add your credentials:
```env
OPENAI_API_KEY=sk-your-key-here
MONGODB_URI=mongodb://localhost:27017  # or your Atlas URI
```

## MongoDB Setup

### Option 1: Local MongoDB
```bash
# Install MongoDB (Ubuntu/Debian)
sudo apt-get install mongodb

# Start MongoDB
sudo systemctl start mongodb
```

### Option 2: MongoDB Atlas (Recommended)

1. Sign up at https://www.mongodb.com/cloud/atlas
2. Create a free cluster
3. Create a database user
4. Get your connection string
5. Create a vector search index with this configuration:

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

## First Example

Create a file `my_first_rag.py`:

```python
from pathlib import Path
from src.rag_pipeline import RAGPipeline, load_config

# Load configuration
config = load_config(Path("configs/default_config.yaml"))

# Initialize pipeline with context manager
with RAGPipeline(config) as pipeline:
    # Add some documents
    pipeline.add_text("""
        Python is a high-level programming language known for its simplicity.
        It supports multiple programming paradigms and has a large ecosystem.
    """)
    
    pipeline.add_text("""
        Machine learning is a subset of artificial intelligence that enables
        systems to learn and improve from experience without being explicitly
        programmed.
    """)
    
    # Query the pipeline
    result = pipeline.query("What is Python?")
    
    print("Question:", result['question'])
    print("\nAnswer:", result['answer'])
    
    # Print sources
    if 'sources' in result:
        print("\nSources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n{i}. {source['content'][:100]}...")
```

Run it:
```bash
python my_first_rag.py
```

## Running Examples

The repository includes three example scripts:

```bash
# Basic usage
python examples/basic_usage.py

# Document processing
python examples/document_processing.py

# Custom configuration
python examples/custom_config.py
```

## Common Tasks

### Add Documents from Files

```python
from pathlib import Path

file_paths = [
    Path("data/doc1.txt"),
    Path("data/doc2.txt"),
]

pipeline.add_text_files(file_paths, metadata={"source": "my_docs"})
```

### Similarity Search (without LLM)

```python
docs = pipeline.similarity_search("machine learning", k=3)
for doc in docs:
    print(doc.page_content)
```

### Custom Prompt Template

```python
custom_prompt = """
Answer the question based on the context below. Be concise.

Context: {context}

Question: {question}

Answer:
"""

result = pipeline.query(
    "What is Python?",
    prompt_template=custom_prompt
)
```

### Programmatic Configuration

```python
from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.rag_pipeline.models import (
    MongoDBConfig,
    EmbeddingConfig,
    LLMConfig,
)

config = RAGPipelineConfig(
    pipeline_name="my_pipeline",
    mongodb=MongoDBConfig(
        connection_string="mongodb://localhost:27017",
        database_name="my_database",
    ),
    embedding=EmbeddingConfig(
        model_name="text-embedding-3-small",
    ),
    llm=LLMConfig(
        model_name="gpt-4o-mini",
        temperature=0.5,
    ),
)

pipeline = RAGPipeline(config)
```

## Testing Your Setup

Run the tests to verify everything is working:

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/ -v
```

## Troubleshooting

### MongoDB Connection Issues

```python
# Test MongoDB connection
from pymongo import MongoClient

client = MongoClient("your-connection-string")
print(client.server_info())  # Should print MongoDB version info
```

### OpenAI API Issues

```python
# Test OpenAI API
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(api_key="your-key")
result = embeddings.embed_query("test")
print(f"Embedding dimension: {len(result)}")
```

### Import Errors

Make sure you're in the virtual environment and have installed all dependencies:

```bash
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. Read the full [README.md](README.md) for detailed documentation
2. Check [IMPLEMENTATION.md](IMPLEMENTATION.md) for architecture details
3. Explore the examples in the `examples/` directory
4. Customize the configuration in `configs/default_config.yaml`
5. Add your own documents and start building!

## Getting Help

- Check the [README.md](README.md) for detailed documentation
- Review the example scripts in `examples/`
- Open an issue on GitHub for bugs or questions

## Configuration Options

See `configs/default_config.yaml` for all available options:

- **MongoDB**: connection, database, collection settings
- **Embeddings**: provider, model, dimensions
- **LLM**: provider, model, temperature, max tokens
- **Retriever**: search type, number of results, score threshold
- **Chunking**: chunk size, overlap, separators
- **Logging**: enable/disable, log level

Happy building! 🚀
