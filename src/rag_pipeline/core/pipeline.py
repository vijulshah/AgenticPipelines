"""Main RAG pipeline implementation."""

import logging
from pathlib import Path
from typing import List, Optional

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from ..models.config import RAGPipelineConfig
from ..utils.document_processor import DocumentProcessor
from ..utils.logger import setup_logger
from .embeddings import EmbeddingsManager
from .llm import LLMManager
from .vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Main RAG pipeline for question-answering."""

    DEFAULT_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, config: RAGPipelineConfig):
        """Initialize RAG pipeline.

        Args:
            config: RAG pipeline configuration.
        """
        self.config = config

        # Set up logging
        if config.enable_logging:
            log_file = Path("logs") / f"{config.pipeline_name}.log"
            self.logger = setup_logger(
                config.pipeline_name,
                level=config.log_level,
                log_file=log_file,
            )
        else:
            self.logger = logging.getLogger(config.pipeline_name)

        self.logger.info(f"Initializing RAG pipeline: {config.pipeline_name}")

        # Initialize components
        self.embeddings_manager = EmbeddingsManager(config.embedding)
        self.llm_manager = LLMManager(config.llm)
        self.document_processor = DocumentProcessor(config.chunking)

        # Initialize vector store
        embeddings = self.embeddings_manager.get_embeddings()
        self.vector_store_manager = VectorStoreManager(config.mongodb, embeddings)

        # Initialize QA chain (lazy initialization)
        self._qa_chain: Optional[RetrievalQA] = None

        self.logger.info("RAG pipeline initialization complete")

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add.

        Returns:
            List of document IDs.
        """
        self.logger.info(f"Adding {len(documents)} documents to pipeline")
        # Chunk documents
        chunked_docs = self.document_processor.chunk_documents(documents)
        self.logger.info(f"Created {len(chunked_docs)} chunks from documents")

        # Add to vector store
        return self.vector_store_manager.add_documents(chunked_docs)

    def add_text_files(
        self, file_paths: List[Path], metadata: Optional[dict] = None
    ) -> List[str]:
        """Load and add text files to the vector store.

        Args:
            file_paths: List of file paths to add.
            metadata: Optional metadata to attach to all documents.

        Returns:
            List of document IDs.
        """
        self.logger.info(f"Processing {len(file_paths)} text files")
        documents = self.document_processor.process_text_files(file_paths, metadata)
        return self.vector_store_manager.add_documents(documents)

    def add_text(self, text: str, metadata: Optional[dict] = None) -> List[str]:
        """Add raw text to the vector store.

        Args:
            text: Raw text content.
            metadata: Optional metadata to attach to the document.

        Returns:
            List of document IDs.
        """
        self.logger.info("Processing raw text content")
        documents = self.document_processor.process_text_content(text, metadata)
        return self.vector_store_manager.add_documents(documents)

    def get_retriever(self) -> BaseRetriever:
        """Get retriever for the vector store.

        Returns:
            Retriever instance.
        """
        vector_store = self.vector_store_manager.get_vector_store()
        search_kwargs = {"k": self.config.retriever.k}

        if self.config.retriever.score_threshold is not None:
            search_kwargs["score_threshold"] = self.config.retriever.score_threshold

        return vector_store.as_retriever(
            search_type=self.config.retriever.search_type,
            search_kwargs=search_kwargs,
        )

    def get_qa_chain(
        self, prompt_template: Optional[str] = None
    ) -> RetrievalQA:
        """Get or create QA chain.

        Args:
            prompt_template: Optional custom prompt template.

        Returns:
            RetrievalQA chain instance.
        """
        if self._qa_chain is None:
            self.logger.info("Creating QA chain")

            llm = self.llm_manager.get_llm()
            retriever = self.get_retriever()

            template = prompt_template or self.DEFAULT_PROMPT_TEMPLATE
            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"],
            )

            self._qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt},
            )

            self.logger.info("QA chain created successfully")

        return self._qa_chain

    def query(
        self,
        question: str,
        return_sources: bool = True,
        prompt_template: Optional[str] = None,
    ) -> dict:
        """Query the RAG pipeline.

        Args:
            question: Question to ask.
            return_sources: Whether to return source documents.
            prompt_template: Optional custom prompt template.

        Returns:
            Dictionary containing answer and optionally source documents.
        """
        self.logger.info(f"Processing query: {question}")

        qa_chain = self.get_qa_chain(prompt_template)
        result = qa_chain.invoke({"query": question})

        response = {
            "question": question,
            "answer": result["result"],
        }

        if return_sources and "source_documents" in result:
            response["sources"] = [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in result["source_documents"]
            ]

        self.logger.info("Query processed successfully")
        return response

    def similarity_search(
        self, query: str, k: Optional[int] = None
    ) -> List[Document]:
        """Perform similarity search without LLM.

        Args:
            query: Search query.
            k: Number of documents to return. Uses config value if not specified.

        Returns:
            List of relevant documents.
        """
        k_value = k or self.config.retriever.k
        self.logger.info(f"Performing similarity search for: {query}")

        return self.vector_store_manager.similarity_search(
            query, k=k_value, score_threshold=self.config.retriever.score_threshold
        )

    def close(self) -> None:
        """Close pipeline and cleanup resources."""
        self.logger.info("Closing RAG pipeline")
        self.vector_store_manager.close()
        self.logger.info("RAG pipeline closed")

    def __enter__(self) -> "RAGPipeline":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.close()
