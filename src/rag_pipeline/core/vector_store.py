"""Vector store initialization and management."""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from ..models.config import MongoDBConfig

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """Manages vector store initialization and operations."""

    def __init__(
        self,
        mongodb_config: MongoDBConfig,
        embeddings: Embeddings,
    ):
        """Initialize vector store manager.

        Args:
            mongodb_config: MongoDB configuration.
            embeddings: Embeddings instance.
        """
        self.mongodb_config = mongodb_config
        self.embeddings = embeddings
        self._client: Optional[MongoClient] = None
        self._vector_store: Optional[VectorStore] = None

    def _get_mongo_client(self) -> MongoClient:
        """Get or create MongoDB client.

        Returns:
            MongoClient instance.
        """
        if self._client is None:
            logger.info("Connecting to MongoDB...")
            self._client = MongoClient(self.mongodb_config.connection_string)
            logger.info("Successfully connected to MongoDB")
        return self._client

    def get_vector_store(self) -> VectorStore:
        """Get or create vector store instance.

        Returns:
            VectorStore instance.
        """
        if self._vector_store is None:
            client = self._get_mongo_client()
            collection = client[self.mongodb_config.database_name][
                self.mongodb_config.collection_name
            ]

            logger.info(
                f"Initializing MongoDB Atlas Vector Search with "
                f"database: {self.mongodb_config.database_name}, "
                f"collection: {self.mongodb_config.collection_name}"
            )

            self._vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self.embeddings,
                index_name=self.mongodb_config.index_name,
            )

        return self._vector_store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store.

        Args:
            documents: List of documents to add.

        Returns:
            List of document IDs.
        """
        vector_store = self.get_vector_store()
        logger.info(f"Adding {len(documents)} documents to vector store...")
        doc_ids = vector_store.add_documents(documents)
        logger.info(f"Successfully added {len(doc_ids)} documents")
        return doc_ids

    def similarity_search(
        self, query: str, k: int = 4, score_threshold: Optional[float] = None
    ) -> List[Document]:
        """Perform similarity search.

        Args:
            query: Search query.
            k: Number of documents to return.
            score_threshold: Optional score threshold.

        Returns:
            List of relevant documents.
        """
        vector_store = self.get_vector_store()

        if score_threshold is not None:
            results = vector_store.similarity_search_with_score(query, k=k)
            # Filter by score threshold
            filtered_results = [
                doc for doc, score in results if score >= score_threshold
            ]
            return filtered_results
        else:
            return vector_store.similarity_search(query, k=k)

    def close(self) -> None:
        """Close MongoDB connection."""
        if self._client is not None:
            logger.info("Closing MongoDB connection...")
            self._client.close()
            self._client = None
            self._vector_store = None
            logger.info("MongoDB connection closed")
