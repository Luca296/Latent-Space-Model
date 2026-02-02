"""
Document ingestion and chunking.
Loads markdown files and splits them into chunks for embedding.
"""

import os
import re
import glob
from dataclasses import dataclass
from typing import List
from config import Config


@dataclass
class Document:
    """Represents a document with metadata."""
    content: str
    source: str
    title: str = ""
    section: str = ""
    start_line: int = 0
    end_line: int = 0


@dataclass
class Chunk:
    """Represents a chunk of a document."""
    text: str
    source: str
    title: str
    section: str
    start_line: int
    end_line: int
    chunk_id: str


def load_documents(corpus_dir: str = None) -> List[Document]:
    """
    Read all .md files from the corpus directory.

    Args:
        corpus_dir: Directory containing markdown files. Defaults to Config.CORPUS_DIR.

    Returns:
        List of Document objects
    """
    if corpus_dir is None:
        corpus_dir = Config.CORPUS_DIR

    documents = []
    md_files = glob.glob(os.path.join(corpus_dir, "*.md"))

    for file_path in md_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        source = os.path.basename(file_path)
        title = extract_title(content) or source

        # Parse sections based on headers
        sections = parse_sections(content, source, title)
        documents.extend(sections)

    return documents


def extract_title(content: str) -> str:
    """Extract the title from markdown frontmatter or first heading."""
    # Try frontmatter
    frontmatter_match = re.search(r'^---\n.*?title:\s*["\']?([^\n"\']+)["\']?\n.*?---', content, re.DOTALL)
    if frontmatter_match:
        return frontmatter_match.group(1).strip()

    # Try first H1
    h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
    if h1_match:
        return h1_match.group(1).strip()

    return ""


def parse_sections(content: str, source: str, title: str) -> List[Document]:
    """
    Parse a markdown document into sections based on headers.

    Args:
        content: The markdown content
        source: The source file name
        title: The document title

    Returns:
        List of Document objects, one per section
    """
    lines = content.split('\n')
    sections = []

    # Track current section
    current_section = "Introduction"
    current_content = []
    start_line = 0

    for i, line in enumerate(lines):
        # Check if this is a header
        header_match = re.match(r'^(#{1,3})\s+(.+)$', line)

        if header_match:
            # Save previous section if it has content
            if current_content:
                sections.append(Document(
                    content='\n'.join(current_content).strip(),
                    source=source,
                    title=title,
                    section=current_section,
                    start_line=start_line + 1,
                    end_line=i
                ))

            # Start new section
            current_section = header_match.group(2).strip()
            current_content = [line]
            start_line = i
        else:
            current_content.append(line)

    # Add final section
    if current_content:
        sections.append(Document(
            content='\n'.join(current_content).strip(),
            source=source,
            title=title,
            section=current_section,
            start_line=start_line + 1,
            end_line=len(lines)
        ))

    return sections


def chunk_document(doc: Document, chunk_size: int = None, overlap: int = None) -> List[Chunk]:
    """
    Split a document into overlapping chunks.

    Args:
        doc: The Document to chunk
        chunk_size: Target chunk size in characters
        overlap: Number of characters to overlap between chunks

    Returns:
        List of Chunk objects
    """
    if chunk_size is None:
        chunk_size = Config.CHUNK_SIZE
    if overlap is None:
        overlap = Config.CHUNK_OVERLAP

    chunks = []
    content = doc.content

    # Simple character-based chunking with overlap
    start = 0
    chunk_num = 0

    while start < len(content):
        end = min(start + chunk_size, len(content))

        # Try to find a good breaking point (end of paragraph or sentence)
        if end < len(content):
            # Look for paragraph break
            para_break = content.rfind('\n\n', start, end)
            if para_break != -1 and para_break > start + chunk_size // 2:
                end = para_break
            else:
                # Look for sentence break
                sent_break = content.rfind('. ', start, end)
                if sent_break != -1 and sent_break > start + chunk_size // 2:
                    end = sent_break + 1

        chunk_text = content[start:end].strip()

        if chunk_text:
            chunks.append(Chunk(
                text=chunk_text,
                source=doc.source,
                title=doc.title,
                section=doc.section,
                start_line=doc.start_line,
                end_line=doc.end_line,
                chunk_id=f"{doc.source}_{doc.section}_{chunk_num}"
            ))

        start = end - overlap if end < len(content) else len(content)
        chunk_num += 1

    return chunks


def ingest_all(corpus_dir: str = None) -> List[Chunk]:
    """
    Process all corpus files and return chunked documents.

    Args:
        corpus_dir: Directory containing markdown files

    Returns:
        List of all chunks from all documents
    """
    documents = load_documents(corpus_dir)
    all_chunks = []

    for doc in documents:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)

    return all_chunks
