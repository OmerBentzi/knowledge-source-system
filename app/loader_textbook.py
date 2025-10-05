"""
Loader for ML/NLP textbook chapters (Option B).

Steps:
- Parse PDF pages (text + OCR images)
- Extract textbook-focused sections (definitions, examples, theorems)
- Chunk text and build index
"""

from __future__ import annotations

from typing import Dict, List

from .common import (
    PageContent,
    build_index,
    chunk_text,
    extract_sections_textbook,
    parse_pdf,
    pages_to_text,
    fetch_url_text,
)


def load_textbook_chapter(source: str) -> Dict:
    if source.lower().startswith("http://") or source.lower().startswith("https://"):
        full_text = fetch_url_text(source)
    else:
        pages: List[PageContent] = parse_pdf(source, ocr_images=True)
        full_text = pages_to_text(pages)
    sections = extract_sections_textbook(full_text)

    base_chunks = chunk_text(full_text, chunk_size=900, overlap=250)

    section_chunks: List[str] = []
    for name, content in sections.items():
        if not content:
            continue
        for c in chunk_text(content, chunk_size=900, overlap=200):
            section_chunks.append(f"[{name}]\n{c}")

    chunks = base_chunks + section_chunks
    metadata = [{"source": "textbook", "i": i} for i in range(len(chunks))]
    return build_index(chunks, metadata)


