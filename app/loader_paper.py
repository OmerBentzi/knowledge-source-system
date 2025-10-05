"""
Loader for research papers (Option C).

Steps:
- Parse PDF pages (text + OCR images)
- Extract paper-focused sections (abstract, method, results)
- Chunk text and build index
"""

from __future__ import annotations

from typing import Dict, List

from .common import (
    PageContent,
    build_index,
    chunk_text,
    extract_sections_paper,
    extract_claims_and_evidence_paper,
    parse_pdf,
    pages_to_text,
    fetch_url_text,
)


def _normalize_arxiv_url(source: str) -> str:
    # Convert arXiv abs to PDF when possible
    # https://arxiv.org/abs/xxxx -> https://arxiv.org/pdf/xxxx.pdf
    lower = source.lower()
    if lower.startswith("https://arxiv.org/abs/"):
        suffix = source.rsplit("/", 1)[-1]
        return f"https://arxiv.org/pdf/{suffix}.pdf"
    return source


def load_research_paper(source: str) -> Dict:
    source = _normalize_arxiv_url(source)
    if source.lower().startswith("http://") or source.lower().startswith("https://"):
        full_text = fetch_url_text(source)
    else:
        pages: List[PageContent] = parse_pdf(source, ocr_images=True)
        full_text = pages_to_text(pages)
    sections = extract_sections_paper(full_text)

    base_chunks = chunk_text(full_text, chunk_size=900, overlap=250)

    section_chunks: List[str] = []
    for name, content in sections.items():
        if not content:
            continue
        for c in chunk_text(content, chunk_size=900, overlap=200):
            section_chunks.append(f"[{name}]\n{c}")

    # Claims/Evidence augmentation for better retrieval
    pairs = extract_claims_and_evidence_paper(sections)
    claim_chunks: List[str] = []
    for p in pairs:
        block = "[claim_evidence]\nClaim: " + p.get("claim", "")
        ev = p.get("evidence", "")
        if ev:
            block += "\nEvidence: " + ev
        claim_chunks.append(block)

    chunks = base_chunks + section_chunks + claim_chunks
    metadata = [{"source": "paper", "i": i} for i in range(len(chunks))]
    return build_index(chunks, metadata)


