"""
CLI entry point for Knowledge Source Processing System.

Usage:
python -m app.main --type [api|textbook|paper] --pdf path/to/file.pdf --question "..." [--use-llm]
"""

from __future__ import annotations

import argparse
import os
from typing import List
import urllib.parse
import urllib.request

from .loader_api import load_api_document
from .loader_textbook import load_textbook_chapter
from .loader_paper import load_research_paper
from .retriever import TfidfRetriever
from .common import read_json, write_json, fetch_url_text


def maybe_answer_with_llm(context: str, question: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    system_prompt = (
        "You are a helpful assistant that answers based on the provided context.\n"
        "Cite relevant snippets. If context is insufficient, say so."
    )
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer concisely."
    )
    # Try modern SDK first
    try:
        from openai import OpenAI  # type: ignore

        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return resp.choices[0].message.content or ""
    except Exception:
        pass

    # Fallback to legacy API if available
    try:
        import openai  # type: ignore

        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=400,
        )
        return resp["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Knowledge Source Processing System")
    parser.add_argument("--type", required=True, choices=["api", "textbook", "paper"], help="Document type")
    parser.add_argument("--pdf", required=False, help="Path to PDF file (or leave empty if using --url)")
    parser.add_argument("--url", required=False, help="URL to fetch HTML for RAG (non-PDF)")
    parser.add_argument("--question", required=True, help="User question")
    parser.add_argument("--use-llm", action="store_true", help="Use OpenAI LLM if OPENAI_API_KEY is set")
    args = parser.parse_args()

    if args.type == "api":
        index_path = os.path.join(os.path.dirname(__file__), "..", "data", "api_index.json")
        loader = load_api_document
    elif args.type == "textbook":
        index_path = os.path.join(os.path.dirname(__file__), "..", "data", "textbook_index.json")
        loader = load_textbook_chapter
    else:
        index_path = os.path.join(os.path.dirname(__file__), "..", "data", "paper_index.json")
        loader = load_research_paper

    # Select source: prefer --url if provided; else --pdf
    source_arg = args.url or args.pdf
    if not source_arg:
        print("Please provide either --url (HTML page) or --pdf (file path or PDF URL).")
        return

    # If a URL is provided and appears to be a PDF, download first; else pass URL directly to loaders
    parsed = urllib.parse.urlparse(source_arg)
    if parsed.scheme in {"http", "https"} and (parsed.path.lower().endswith(".pdf")):
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        os.makedirs(data_dir, exist_ok=True)
        filename = os.path.basename(parsed.path) or "downloaded.pdf"
        dest_path = os.path.join(data_dir, filename)
        try:
            urllib.request.urlretrieve(source_arg, dest_path)
            source_arg = dest_path
        except Exception as e:
            print(f"Failed to download PDF from URL: {e}")
            return

    # Build index fresh for each run (simple approach) and save
    index = loader(source_arg)
    write_json(index_path, index)

    # Retrieval
    retriever = TfidfRetriever()
    retriever.fit(index)
    results = retriever.query(args.question, top_k=5)

    if not results:
        print("No relevant results found by retriever.")
        if args.use_llm:
            # Try LLM directly using raw page text in URL mode
            if args.url:
                raw_text = fetch_url_text(args.url) or ""
                if raw_text.strip():
                    llm_answer = maybe_answer_with_llm(raw_text[:12000], args.question)
                    if llm_answer:
                        print("=== Generative answer (LLM) ===\n")
                        print(llm_answer)
                        return
            print("LLM not available or insufficient page text. Consider passing a specific subpage or a PDF.")
        return

    print("Top matches (local extractive mode):\n")
    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r.score:.3f}\n{r.chunk[:600]}\n")

    if args.use_llm:
        # If URL mode and scores are extremely low, try raw page text fallback
        max_score = max((r.score for r in results), default=0.0)
        if args.url and max_score < 1e-3:
            raw_text = fetch_url_text(args.url) or ""
            if raw_text.strip():
                llm_answer = maybe_answer_with_llm(raw_text[:12000], args.question)
                if llm_answer:
                    print("=== Generative answer (LLM) — URL fallback ===\n")
                    print(llm_answer)
                    return

        # Otherwise combine top chunks for LLM context
        combined_context = "\n\n".join(r.chunk for r in results)
        llm_answer = maybe_answer_with_llm(combined_context, args.question)
        if llm_answer:
            print("=== Generative answer (LLM) ===\n")
            print(llm_answer)
        else:
            print("LLM not available or failed. Showing local results only.")


if __name__ == "__main__":
    main()


