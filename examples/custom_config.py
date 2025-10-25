"""Custom configuration example."""

from src.rag_pipeline import RAGPipeline, RAGPipelineConfig
from src.rag_pipeline.models import (
    ChunkingConfig,
    EmbeddingConfig,
    LLMConfig,
    MongoDBConfig,
    RetrieverConfig,
)


def main() -> None:
    """Run custom configuration example."""
    # Create custom configuration programmatically
    config = RAGPipelineConfig(
        pipeline_name="custom_rag_pipeline",
        mongodb=MongoDBConfig(
            connection_string="mongodb://localhost:27017",
            database_name="custom_rag_db",
            collection_name="custom_vectors",
            index_name="custom_index",
        ),
        embedding=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",
            dimensions=1536,
        ),
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-4o-mini",
            temperature=0.5,
            max_tokens=500,
        ),
        retriever=RetrieverConfig(
            search_type="similarity",
            k=3,
            score_threshold=0.7,
        ),
        chunking=ChunkingConfig(
            chunk_size=800,
            chunk_overlap=150,
            separator="\n",
        ),
        enable_logging=True,
        log_level="DEBUG",
    )

    # Initialize pipeline with custom config
    with RAGPipeline(config) as pipeline:
        # Add custom prompt template
        custom_prompt = """You are a helpful AI assistant. Use the following context
to answer the question. Be concise and accurate.

Context:
{context}

Question: {question}

Concise Answer:"""

        # Add sample data
        text = """
        FastAPI is a modern, fast web framework for building APIs with Python
        based on standard Python type hints. It's designed to be easy to use
        and high-performance, comparable to NodeJS and Go.

        FastAPI automatically generates interactive API documentation using
        Swagger UI and ReDoc. It provides automatic data validation using
        Pydantic models.
        """

        pipeline.add_text(text, metadata={"topic": "FastAPI"})

        # Query with custom prompt
        result = pipeline.query(
            "What are the key features of FastAPI?",
            prompt_template=custom_prompt,
        )

        print(f"Question: {result['question']}")
        print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
