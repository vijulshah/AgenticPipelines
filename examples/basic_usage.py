"""Basic RAG pipeline example."""

from pathlib import Path

from src.rag_pipeline import RAGPipeline, load_config


def main() -> None:
    """Run basic RAG pipeline example."""
    # Load configuration
    config_path = Path("configs/default_config.yaml")
    config = load_config(config_path)

    # Initialize pipeline
    with RAGPipeline(config) as pipeline:
        # Add some sample text
        sample_text = """
        Artificial Intelligence (AI) is the simulation of human intelligence 
        processes by machines, especially computer systems. These processes 
        include learning, reasoning, and self-correction.
        
        Machine Learning is a subset of AI that provides systems the ability 
        to automatically learn and improve from experience without being 
        explicitly programmed.
        
        Deep Learning is a subset of machine learning that uses neural networks 
        with multiple layers. These neural networks attempt to simulate the 
        behavior of the human brain to learn from large amounts of data.
        """

        print("Adding documents to vector store...")
        pipeline.add_text(sample_text, metadata={"source": "AI basics"})

        # Query the pipeline
        print("\nQuerying the RAG pipeline...")
        question = "What is Machine Learning?"
        result = pipeline.query(question)

        print(f"\nQuestion: {result['question']}")
        print(f"\nAnswer: {result['answer']}")

        if "sources" in result:
            print("\nSources:")
            for i, source in enumerate(result["sources"], 1):
                print(f"\n{i}. {source['content'][:200]}...")
                print(f"   Metadata: {source['metadata']}")


if __name__ == "__main__":
    main()
