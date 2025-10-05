"""
Common utilities for the Knowledge Source Processing System.

Responsibilities:
- Load and parse PDFs using PyMuPDF (text + embedded images)
- OCR images using Tesseract (pytesseract)
- Chunk text into overlapping windows
- Lightweight section extraction heuristics per document type
- JSON index read/write helpers
- Optional environment loading via python-dotenv

This module is intentionally dependency-light and defensive: if OCR is
unavailable or an image fails, we continue gracefully.
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from urllib import request, error as urlerror

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # optional; only needed for --url HTML mode

def _render_with_playwright(url: str, timeout_ms: int = 15000) -> str:
    """Render JS-heavy pages with Playwright (if available) and return HTML."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore
    except Exception:
        return ""
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout_ms)
            page.wait_for_load_state("networkidle", timeout=timeout_ms)
            html = page.content()
            browser.close()
            return html
    except Exception:
        return ""

try:
    import fitz  # PyMuPDF
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyMuPDF (fitz) is required. Please install pymupdf.") from exc

try:
    from PIL import Image, ImageFilter, ImageOps
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Pillow is required. Please install pillow.") from exc

try:
    import pytesseract
except Exception:
    pytesseract = None  # Graceful degradation if Tesseract is not available

try:  # optional
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Honor explicit Tesseract path if provided via env
try:
    if pytesseract is not None:
        tess_cmd = os.getenv("TESSERACT_CMD")
        if tess_cmd:
            # type: ignore[attr-defined]
            pytesseract.pytesseract.tesseract_cmd = tess_cmd
except Exception:
    pass


@dataclass
class PageContent:
    page_number: int
    text: str


def _extract_text_from_page(page: "fitz.Page") -> str:
    text = page.get_text("text") or ""
    return text.strip()


def _extract_images_from_page(page: "fitz.Page") -> List[Image.Image]:
    images: List[Image.Image] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image.get("image")
            if not image_bytes:
                continue
            pil_image = Image.open(io.BytesIO(image_bytes))
            images.append(pil_image)
        except Exception:
            # Skip problematic images and continue
            continue
    return images


def _ocr_image(img: Image.Image) -> str:
    if pytesseract is None:
        return ""  # OCR not available; handled by caller
    try:
        # Preprocess: upscale, grayscale, autocontrast, sharpen, light binarization
        w, h = img.size
        scale = 3 if max(w, h) < 1500 else 2
        if scale > 1:
            img = img.resize((w * scale, h * scale))
        gray = ImageOps.grayscale(img)
        gray = ImageOps.autocontrast(gray)
        gray = gray.filter(ImageFilter.SHARPEN)
        bw = gray.point(lambda x: 255 if x > 180 else 0, mode='1')

        lang = os.getenv("OCR_LANG", "eng")
        for psm in (6, 4, 11):
            config = f"--psm {psm} --oem 3"
            text = pytesseract.image_to_string(bw, config=config, lang=lang)
            text = (text or "").strip()
            if len(text) >= 10:
                return text
        # Fallback single try on grayscale
        text = pytesseract.image_to_string(gray, config="--psm 6 --oem 3", lang=lang)
        return (text or "").strip()
    except Exception:
        return ""


def parse_pdf(pdf_path: str, ocr_images: bool = True) -> List[PageContent]:
    """Parse a PDF into per-page text, performing OCR on embedded images.

    Args:
        pdf_path: Path to the PDF file
        ocr_images: Whether to attempt OCR on embedded images

    Returns:
        List of PageContent with combined text per page.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: List[PageContent] = []
    for i in range(len(doc)):
        page = doc[i]
        page_text = _extract_text_from_page(page)
        if ocr_images:
            image_texts: List[str] = []
            for img in _extract_images_from_page(page):
                ocr_text = _ocr_image(img)
                if ocr_text:
                    image_texts.append(ocr_text)
            if image_texts:
                page_text = (page_text + "\n\n" + "\n".join(image_texts)).strip()

            # Fallback: rasterize full page for OCR if text is missing/very short
            if (not page_text or len(page_text) < 20) and pytesseract is not None:
                try:
                    zoom = 3.0  # stronger upscale for better OCR
                    mat = fitz.Matrix(zoom, zoom)
                    pix = page.get_pixmap(matrix=mat)
                    pil = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    full_ocr = _ocr_image(pil)
                    if full_ocr:
                        page_text = (page_text + "\n\n" + full_ocr).strip()
                except Exception:
                    pass

        pages.append(PageContent(page_number=i + 1, text=page_text))

    return pages


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[str]:
    """Split text into overlapping chunks by whitespace.

    Uses a simple token surrogate (characters) for robustness.
    """
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    tokens = list(text)
    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for start in range(0, len(tokens), step):
        end = min(len(tokens), start + chunk_size)
        slice_tokens = tokens[start:end]
        chunk = "".join(slice_tokens).strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(tokens):
            break
    return chunks


def extract_sections_api(text: str) -> Dict[str, str]:
    """Heuristically extract sections from API documentation text.

    Looks for headings like 'Endpoint', 'Parameters', 'Authentication'.
    """
    sections = {
        "endpoints": [],
        "parameters": [],
        "authentication": [],
        "other": [],
    }
    current = "other"
    for line in text.splitlines():
        lower = line.lower().strip()
        if any(k in lower for k in ["endpoint", "url", "route"]):
            current = "endpoints"
        elif any(k in lower for k in ["param", "parameter", "payload", "body"]):
            current = "parameters"
        elif any(k in lower for k in ["auth", "token", "oauth", "apikey", "api key"]):
            current = "authentication"
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_sections_textbook(text: str) -> Dict[str, str]:
    """Heuristically extract definitions, examples, theorems from textbook-like text."""
    sections = {
        "definitions": [],
        "examples": [],
        "theorems": [],
        "other": [],
    }
    current = "other"
    for line in text.splitlines():
        lower = line.lower().strip()
        if lower.startswith("definition") or "def." in lower:
            current = "definitions"
        elif lower.startswith("example") or lower.startswith("examples"):
            current = "examples"
        elif lower.startswith("theorem") or "lemma" in lower or "proposition" in lower:
            current = "theorems"
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_sections_paper(text: str) -> Dict[str, str]:
    """Heuristically extract abstract, method, results sections from paper text."""
    sections = {
        "abstract": [],
        "method": [],
        "results": [],
        "other": [],
    }
    current = "other"
    for line in text.splitlines():
        lower = line.lower().strip()
        if lower.startswith("abstract"):
            current = "abstract"
        elif any(lower.startswith(k) for k in ["method", "methods", "approach"]):
            current = "method"
        elif any(lower.startswith(k) for k in ["result", "results", "experiments", "evaluation"]):
            current = "results"
        sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_claims_and_evidence_paper(sections: Dict[str, str]) -> List[Dict[str, str]]:
    """Very lightweight claim/evidence mining for research papers.

    Heuristics:
    - Claims: sentences in abstract/method containing keywords (propose, introduce, present, we)
    - Evidence: sentences in results containing (improve, achieve, outperform, accuracy, f1, bleu)
    Returns list of {claim, evidence} pairs (evidence may be empty).
    """
    import re

    def split_sentences(block: str) -> List[str]:
        # Simple sentence split on period/question/exclamation
        parts = re.split(r"(?<=[.!?])\s+", block.strip())
        return [p.strip() for p in parts if p.strip()]

    abstract = sections.get("abstract", "")
    method = sections.get("method", "")
    results = sections.get("results", "")

    claim_kws = ["we propose", "we introduce", "we present", "our method", "this paper"]
    ev_kws = [
        "improve", "improves", "improved", "achieve", "achieves", "achieved",
        "outperform", "outperforms", "outperformed", "accuracy", "f1", "bleu", "rouge",
        "results", "significant", "state-of-the-art", "sota"
    ]

    claim_sents: List[str] = []
    for s in split_sentences(abstract) + split_sentences(method):
        lower = s.lower()
        if any(k in lower for k in claim_kws):
            claim_sents.append(s)

    evidence_sents: List[str] = []
    for s in split_sentences(results):
        lower = s.lower()
        if any(k in lower for k in ev_kws):
            evidence_sents.append(s)

    pairs: List[Dict[str, str]] = []
    if not claim_sents:
        return pairs
    if not evidence_sents:
        for c in claim_sents:
            pairs.append({"claim": c, "evidence": ""})
        return pairs

    # Greedy pair: align in order
    for i, c in enumerate(claim_sents):
        e = evidence_sents[min(i, len(evidence_sents) - 1)]
        pairs.append({"claim": c, "evidence": e})
    return pairs


def write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: List[str], metadata: Optional[List[dict]] = None) -> dict:
    """Build a minimal index structure for downstream retriever.

    Args:
        chunks: list of text chunks
        metadata: per-chunk metadata dicts (e.g., page numbers)
    """
    if metadata is None:
        metadata = [{} for _ in chunks]
    return {"chunks": chunks, "metadata": metadata}


def pages_to_text(pages: List[PageContent]) -> str:
    return "\n\n".join(f"[Page {p.page_number}]\n{p.text}" for p in pages if p.text)


def fetch_url_text(url: str) -> str:
    """Fetch HTML content from a URL and return visible text.

    If BeautifulSoup isn't available, returns a naive decoded body.
    """
    try:
        req = request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/127.0.0.0 Safari/537.36"
                )
            },
        )
        with request.urlopen(req) as resp:
            content_type = resp.headers.get("Content-Type", "").lower()
            raw = resp.read()
    except Exception:
        raw = b""
        content_type = ""

    # If appears to be a PDF, return empty (let caller handle as PDF path flow)
    if "application/pdf" in content_type:
        return ""

    body = ""
    if raw:
        try:
            body = raw.decode("utf-8", errors="ignore")
        except Exception:
            body = raw.decode(errors="ignore")
    if not body:
        # Fallback to JS rendering if initial fetch failed or empty
        body = _render_with_playwright(url)
        if not body:
            return ""

    if BeautifulSoup is None:
        return body

    try:
        soup = BeautifulSoup(body, "html.parser")
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
            tag.extract()
        text = soup.get_text("\n")
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join([l for l in lines if l])
        return text
    except Exception:
        return body


