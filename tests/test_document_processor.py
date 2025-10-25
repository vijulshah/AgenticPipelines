"""Tests for document processor."""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.rag_pipeline.models.config import ChunkingConfig
from src.rag_pipeline.utils.document_processor import DocumentProcessor


class TestDocumentProcessor:
    """Test document processor."""

    @pytest.fixture
    def processor(self) -> DocumentProcessor:
        """Create a document processor instance."""
        config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=20,
            separator="\n\n",
        )
        return DocumentProcessor(config)

    @pytest.fixture
    def sample_text_file(self, tmp_path: Path) -> Path:
        """Create a sample text file."""
        file_path = tmp_path / "sample.txt"
        content = "This is a test document.\n\nIt has multiple paragraphs."
        file_path.write_text(content)
        return file_path

    def test_load_text_file(
        self, processor: DocumentProcessor, sample_text_file: Path
    ) -> None:
        """Test loading a text file."""
        doc = processor.load_text_file(sample_text_file)
        assert isinstance(doc, Document)
        assert "test document" in doc.page_content
        assert doc.metadata["source"] == str(sample_text_file)

    def test_load_text_file_with_metadata(
        self, processor: DocumentProcessor, sample_text_file: Path
    ) -> None:
        """Test loading a text file with custom metadata."""
        metadata = {"author": "test_author"}
        doc = processor.load_text_file(sample_text_file, metadata)
        assert doc.metadata["author"] == "test_author"
        assert doc.metadata["source"] == str(sample_text_file)

    def test_load_nonexistent_file(self, processor: DocumentProcessor) -> None:
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            processor.load_text_file(Path("nonexistent.txt"))

    def test_chunk_documents(self, processor: DocumentProcessor) -> None:
        """Test chunking documents."""
        long_text = "A" * 250  # Text longer than chunk_size
        doc = Document(page_content=long_text)
        chunks = processor.chunk_documents([doc])
        assert len(chunks) > 1  # Should be split into multiple chunks

    def test_process_text_content(self, processor: DocumentProcessor) -> None:
        """Test processing raw text content."""
        text = "This is a test.\n\n" * 20  # Create long text
        chunks = processor.process_text_content(text)
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_process_text_files(
        self, processor: DocumentProcessor, sample_text_file: Path
    ) -> None:
        """Test processing multiple text files."""
        chunks = processor.process_text_files([sample_text_file])
        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
