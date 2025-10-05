## Example Runs

These examples assume you have suitable PDFs:
- `data/sample_api.pdf`
- `data/sample_textbook.pdf`
- `data/sample_paper.pdf`

Commands:
```bash
python -m app.main --type api --pdf data/sample_api.pdf --question "How do I authenticate?"
python -m app.main --type textbook --pdf data/sample_textbook.pdf --question "Provide the definition of cross-entropy."
python -m app.main --type paper --pdf data/sample_paper.pdf --question "Summarize the results"
```

Optional with LLM (requires `OPENAI_API_KEY`):
```bash
python -m app.main --type paper --pdf data/sample_paper.pdf --question "What is the main contribution?" --use-llm
```

Sample outputs (illustrative only):

API Docs Q&A
- Q: How do I authenticate?
  - A (local): Mentions "Authentication" section with token header `Authorization: Bearer <token>`.
- Q: List the available endpoints.
  - A (local): Shows a chunk listing `/v1/users`, `/v1/items`, `/v1/items/{id}`.
- Q: What parameters does the create endpoint accept?
  - A (local): Shows fields like `name`, `description`, `price`.

Textbook Q&A
- Q: Provide the definition of cross-entropy.
  - A (local): Returns the definition lines around "Definition: Cross-Entropy".
- Q: Give a simple example explaining word embeddings.
  - A (local): Returns an "Example" block about distributional semantics.
- Q: State the central theorem referenced in this chapter.
  - A (local): Returns a theorem statement and conditions.

Research Paper Q&A
- Q: Summarize the abstract.
  - A (local): Returns the abstract section with objectives and contributions.
- Q: What is the proposed method?
  - A (local): Returns the method section with model architecture notes.
- Q: Summarize the results.
  - A (local): Returns the results section mentioning datasets and metrics.


