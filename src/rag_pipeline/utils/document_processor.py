"""Document processing utilities."""

from pathlib import Path
from typing import List, Optional

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..models.config import ChunkingConfig


class DocumentProcessor:
    """Handles document loading and chunking."""

    def __init__(self, chunking_config: ChunkingConfig):
        """Initialize document processor.

        Args:
            chunking_config: Configuration for document chunking.
        """
        self.chunking_config = chunking_config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.chunk_overlap,
            separators=[
                chunking_config.separator,
                "\n",
                " ",
                "",
            ],
        )

    def load_text_file(
        self, file_path: Path, metadata: Optional[dict] = None
    ) -> Document:
        """Load a text file as a Document.

        Args:
            file_path: Path to the text file.
            metadata: Optional metadata to attach to the document.

        Returns:
            Document instance.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        doc_metadata = metadata or {}
        doc_metadata["source"] = str(file_path)

        return Document(page_content=content, metadata=doc_metadata)

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks.

        Args:
            documents: List of documents to chunk.

        Returns:
            List of chunked documents.
        """
        return self.text_splitter.split_documents(documents)

    def process_text_files(
        self, file_paths: List[Path], metadata: Optional[dict] = None
    ) -> List[Document]:
        """Load and chunk multiple text files.

        Args:
            file_paths: List of file paths to process.
            metadata: Optional metadata to attach to all documents.

        Returns:
            List of chunked documents.
        """
        documents = []
        for file_path in file_paths:
            doc = self.load_text_file(file_path, metadata)
            documents.append(doc)

        return self.chunk_documents(documents)

    def process_text_content(
        self, text: str, metadata: Optional[dict] = None
    ) -> List[Document]:
        """Process raw text content into chunks.

        Args:
            text: Raw text content.
            metadata: Optional metadata to attach to all documents.

        Returns:
            List of chunked documents.
        """
        doc = Document(page_content=text, metadata=metadata or {})
        return self.chunk_documents([doc])
