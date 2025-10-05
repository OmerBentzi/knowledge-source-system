## Knowledge Source Processing System

A Retrieval-Augmented Generation (RAG) pipeline that processes three document types:
- API/Model documentation
- ML/NLP textbook chapters
- Research papers

It extracts text (and images via OCR for PDFs), builds an index, retrieves relevant chunks with either TF‑IDF or Embeddings, and can optionally generate answers using an LLM. A Streamlit UI provides a conversational experience.

### Features
- PyMuPDF PDF parsing (text + embedded images)
- OCR on images via `pytesseract` (graceful fallback if not installed)
- Light section heuristics per type:
  - API docs: endpoints, parameters, authentication
  - Textbook: definitions, examples, theorems
  - Research paper: abstract, method, results (+ simple claim/evidence pairing)
- Retrievers:
  - TF‑IDF + cosine similarity
  - Embeddings retriever (OpenAI `text-embedding-3-small`) for better web results
- Generative answers with OpenAI `gpt-4o-mini` (optional)
- URL mode: fetches HTML; auto-falls back to Playwright rendering for JS‑heavy sites
- PDF mode: local files or direct PDF URLs; OCR is used for images where possible

### Project Structure
```
knowledge_source_system/
  app/
    __init__.py
    common.py
    retriever.py
    loader_api.py
    loader_textbook.py
    loader_paper.py
    main.py
    ui.py
  data/
    api_index.json
    textbook_index.json
    paper_index.json
  README.md
  requirements.txt
  example_run.md
```

### Requirements
- Python 3.10+
- Windows/macOS/Linux


Install Python deps:
```bash
pip install -r requirements.txt
python -m playwright install chromium  # for JS-rendered pages in URL mode
```

Optional: set OpenAI API key (LLM and Embeddings):
```bash
# Windows PowerShell
setx OPENAI_API_KEY "sk-..."
# macOS/Linux
export OPENAI_API_KEY=sk-...
```

### CLI Usage
From the project root:
```bash
python -m app.main --type paper --pdf path/to/file.pdf --question "Summarize the results"
```

Options:
- `--type`: `api` | `textbook` | `paper`
- `--pdf`: path to a local PDF file or a direct PDF URL
- `--url`: web page URL (HTML; will auto-render if JS-heavy)
- `--question`: your question
- `--use-llm`: enable LLM answer if `OPENAI_API_KEY` is set

Examples:
```bash
# Local extractive answers only (PDF)
python -m app.main --type api --pdf data/sample_api.pdf --question "How do I authenticate?"

# With LLM generation (PDF)
python -m app.main --type paper --pdf data/sample_paper.pdf --question "What is the main contribution?" --use-llm

# URL mode (HTML)
python -m app.main --type api --url "https://platform.openai.com/docs/guides/authentication" \
  --question "Which header and value do I use for auth?" --use-llm
```

### Streamlit Chat UI (optional)
Interactive chat with document selection and retrieval mode controls.

Install/update deps (if not already):
```bash
pip install -r requirements.txt
python -m playwright install chromium
```

Run the UI from project root:
```bash
python -m streamlit run app/ui.py
```

In the UI sidebar:
- Choose `api` | `textbook` | `paper`
- Provide a URL (HTML or direct PDF) or upload a PDF, then click “Build/Refresh Index”
- Paste your `OPENAI_API_KEY` to enable LLM/Embeddings
- Choose Retriever: `TF-IDF` (fast) or `Embeddings` (better for web)
- Adjust `Result threshold` and `Top K` as needed
- Expand “Sources” to see supporting chunks above threshold

Footer: “Made by Omer Ben Simon for Spacial AI”.

### Design Choices
- Parsing: PyMuPDF for robust text extraction; OCR for images when available.
- Chunking: overlapping character windows (simple, language-agnostic).
- Section heuristics: keyword-based routing by document type.
- Retrievers: TF‑IDF (bi-grams) and Embeddings (OpenAI, cosine). Embeddings recommended for web content.
- Generative Mode: If retrieval is weak in URL mode, the system can pass raw page text to the LLM as fallback.
- URL Rendering: Browser-like User-Agent; Playwright headless rendering when static HTML is insufficient.

### Technical Decision Rationale

**Context**: Build a RAG system that processes PDFs and web pages with both text and visual content, supporting three document types with different retrieval needs.

**Decision**: Use TF‑IDF as primary retriever with optional Embeddings fallback, PyMuPDF for parsing, and Tesseract for OCR.

**Why**: 
- TF‑IDF is dependency-light, transparent, and works well for structured documents (API docs, papers)
- Embeddings excel for web content and semantic similarity but require API calls
- PyMuPDF handles both text extraction and embedded image access robustly
- Tesseract provides mature OCR with good preprocessing options
- Character-based chunking avoids tokenization complexity across languages

**Trade-offs**: 
- No persistent vector store (rebuilds each run) — acceptable for demo scope
- Could use FAISS/Pinecone for production scale, but adds complexity
- Section heuristics are simple keyword-based; could use ML models for better accuracy
- OCR quality depends on image resolution; could add advanced preprocessing pipelines

**Alternative approaches considered**:
- LangChain pipeline: More structured but heavier for this scope
- Direct OpenAI embeddings: Simpler but less flexible than dual retriever approach
- FAISS vector store: Better for production but overkill for demo
- Layout-aware parsing: Would improve table/figure handling but requires additional ML models

### 🧩 Multi-Domain Support Rationale

**Decision**: Instead of focusing on a single document type (API docs, textbooks, or papers), this system was intentionally designed to support all three simultaneously through a unified RAG architecture.

**Why**: Each source type represents a distinct knowledge structure:
- **API/Model Docs** → hierarchical, parameter-driven, precise
- **ML/NLP Textbooks** → conceptual, sequential, explanatory  
- **Research Papers** → experimental, evidence-based, analytical

Supporting all of them demonstrates the pipeline's schema-agnostic flexibility and ensures it can generalize across diverse knowledge domains.

**Benefits**:
- **Architectural flexibility**: One modular pipeline handles heterogeneous data (structured, narrative, or scientific)
- **Real-world relevance**: In practical AI systems, users often query across multiple knowledge domains simultaneously (e.g., code + theory + research)
- **Evaluation depth**:

| Document Type | Tests | Demonstrates |
|---------------|-------|--------------|
| API Docs | Hierarchical parsing | Structured retrieval precision |
| Textbooks | Conceptual understanding | Semantic embeddings & reasoning |
| Research Papers | Claim–evidence mapping | Context summarization & multi-hop QA |

**Implementation Highlights**:
- Unified load → split → embed → retrieve → generate pipeline shared across all types
- Lightweight type-specific heuristics to adapt parsing and chunking behavior
- Compatible with both TF-IDF and Embedding-based retrievers for hybrid recall strategies
- Designed for easy scalability: adding new domains (e.g., legal docs or design specs) requires no architecture change—only a new loader

**Impact**: This design mirrors production-grade GenAI systems used in enterprise knowledge assistants. It showcases system design maturity, cross-domain reasoning, and engineering foresight, positioning the project as more than a demo — a foundation for scalable AI knowledge retrieval.

### Limitations
- OCR quality depends on Tesseract and the PDF’s image quality.
- Section detection is heuristic and may miss unusual headings.
- Index is rebuilt each run (no persistent vector store).
- Some websites block scraping; even Playwright may not capture gated content.

### Improvements
- Sentence segmentation and better chunking strategies.
- Vector DB (e.g., FAISS) for persistent semantic search.
- Site crawling for multi-page documentation portals.
- Layout-aware parsing for tables/figures and caption linking.

### 🧩 Tech Stack Rationale

This project was intentionally built **without heavy frameworks** (like LangChain) to demonstrate a strong grasp of core RAG and retrieval mechanics, from data ingestion to query generation.  
Each component was carefully selected for clarity, performance, and maintainability.

| Component | Purpose | Rationale |
|------------|----------|------------|
| **PyMuPDF (`fitz`)** | PDF parsing and text extraction | Robust, fast, and lightweight library supporting text, metadata, and embedded image extraction. Handles complex layouts better than pure text-based PDF parsers. |
| **Tesseract (`pytesseract`)** | OCR for image-based PDFs | Enables processing of scanned or hybrid documents containing visual data (e.g., diagrams, tables). Ensures no loss of context in mixed-media PDFs. |
| **scikit-learn (TF-IDF + cosine similarity)** | Classical retriever | A transparent, dependency-light retrieval method ideal for structured text (API docs, papers). Easy to debug and interpret compared to dense vector models. |
| **OpenAI API (Embeddings + GPT)** | Semantic retrieval and generative response | Complements TF-IDF with semantic embeddings (`text-embedding-3-small`) for web content. LLM integration (`gpt-4o-mini`) enables answer synthesis and reasoning beyond keyword search. |
| **Streamlit** | Interactive front-end | Provides a clean and minimal interface for testing, visualizing retrieval results, and debugging retrieval thresholds. Great for prototyping without a full web stack. |
| **Playwright** | Web rendering engine | Handles JS-heavy or dynamically loaded web pages (common in documentation portals). Ensures that full page content—including interactive sections—is captured before indexing. |

---

### ⚙️ Design Philosophy

The system is **framework-independent and transparent**:
- End-to-end data flow (parsing → indexing → retrieval → generation)
- Interchangeable retrievers (TF-IDF vs. Embeddings)
- Modular architecture allowing future integration with FAISS, Pinecone, or LangChain
- Production-ready foundation that could scale into a true knowledge assistant for technical domains

---

### 🧠 Why Not LangChain?

LangChain is excellent for orchestration, but in a short-form home assignment:
- It adds abstraction layers that obscure the core logic.
- Custom implementations make reasoning easier to explain in interviews.
- The code demonstrates that **you can build LangChain-like functionality from first principles** — a skill highly valued in applied AI roles.

---

### 🚀 Optional Future Enhancements

- Replace TF-IDF with **FAISS or Chroma** for persistent semantic vector stores.
- Add **hybrid search** combining sparse (TF-IDF) and dense (embeddings) retrieval.
- Wrap Streamlit UI in a **FastAPI** backend for deployment.
- Add **Dockerfile + MLflow** for experiment tracking and reproducibility.

---

  

### Sample Q&A Prompts
API Docs:
- "List the available endpoints."
- "What authentication is required?"
- "What parameters does the create endpoint accept?"

Textbook:
- "Provide the definition of cross-entropy."
- "Give a simple example explaining word embeddings."
- "State the central theorem referenced in this chapter."

Research Paper:
- "Summarize the abstract."
- "What is the proposed method?"
- "Summarize the results section."

### Troubleshooting
- “ModuleNotFoundError: No module named app”: run commands from the `knowledge_source_system` directory.
- “No relevant results found / score ≈ 0”: use a text-heavy URL or a PDF; switch Retriever to Embeddings; increase Top‑K and threshold.
- Streamlit: launch with `python -m streamlit run app/ui.py` to avoid PATH issues.


