"""Document loading and processing example."""

from pathlib import Path

from src.rag_pipeline import RAGPipeline, load_config


def main() -> None:
    """Run document processing example."""
    # Load configuration
    config_path = Path("configs/default_config.yaml")
    config = load_config(config_path)

    # Initialize pipeline
    with RAGPipeline(config) as pipeline:
        # Create sample documents
        docs_dir = Path("data/sample_docs")
        docs_dir.mkdir(parents=True, exist_ok=True)

        # Create sample document files
        doc1_path = docs_dir / "python_intro.txt"
        doc1_path.write_text(
            """
            Python is a high-level, interpreted programming language known for 
            its clear syntax and readability. It supports multiple programming 
            paradigms including procedural, object-oriented, and functional 
            programming.
            
            Python's extensive standard library and third-party packages make 
            it suitable for various applications including web development, 
            data analysis, machine learning, and automation.
            """
        )

        doc2_path = docs_dir / "langchain_intro.txt"
        doc2_path.write_text(
            """
            LangChain is a framework for developing applications powered by 
            large language models (LLMs). It provides tools and abstractions 
            for building context-aware and reasoning applications.
            
            LangChain supports various components including chains, agents, 
            memory, and retrieval systems. It integrates with multiple LLM 
            providers and vector stores for RAG applications.
            """
        )

        # Add documents to pipeline
        print("Loading and processing documents...")
        doc_ids = pipeline.add_text_files([doc1_path, doc2_path])
        print(f"Added {len(doc_ids)} document chunks")

        # Perform similarity search
        print("\nPerforming similarity search...")
        query = "What is LangChain?"
        docs = pipeline.similarity_search(query, k=2)

        print(f"\nQuery: {query}")
        print(f"\nFound {len(docs)} relevant documents:")
        for i, doc in enumerate(docs, 1):
            print(f"\n{i}. {doc.page_content[:200]}...")
            print(f"   Metadata: {doc.metadata}")

        # Query with LLM
        print("\n" + "=" * 80)
        print("Querying with LLM...")
        result = pipeline.query("Explain what LangChain is used for")

        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
