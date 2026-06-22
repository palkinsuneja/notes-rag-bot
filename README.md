# Notes Q&A Bot 📚

A RAG-powered study assistant that answers questions from your personal notes with source citations. Built with LangChain, ChromaDB, and Groq.

---

## Project Structure

notes-rag-bot/

├── sample_notes/        ← add your notes here (.txt or .pdf)

│   └── spm_notes.txt    (sample SPM notes for testing)

├── rag_pipeline.py      ← core RAG logic (chunking, embedding, retrieval)

├── app.py               ← terminal version for quick testing

├── streamlit_app.py     ← Streamlit UI (deployable)

├── requirements.txt

└── README.md

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Groq API key

Get a free key at https://console.groq.com

```bash
export GROQ_API_KEY="your-key-here"      # Mac/Linux
set GROQ_API_KEY=your-key-here           # Windows
```

### 3. Add your notes

Drop your `.txt` or `.pdf` files into the `sample_notes/` folder. Sample SPM notes are already included for testing.

---

## Run

### Option A — Terminal (fastest for testing)

```bash
python app.py
```

### Option B — Streamlit UI (full demo)

```bash
python -m streamlit run streamlit_app.py
```

---

## How It Works

| Step | What happens |
|---|---|
| **Load** | Reads all .txt and .pdf files from sample_notes/ |
| **Chunk** | Splits text into 500-char pieces with 80-char overlap |
| **Embed** | Converts each chunk to a 384-dim vector using sentence-transformers |
| **Store** | Saves vectors in ChromaDB locally |
| **Retrieve** | Embeds user query, finds top-3 similar chunks via cosine similarity |
| **Generate** | Sends retrieved context + question to Groq (llama-3.3-70b) for answer |

---

## Things to Experiment With

**1. Change `CHUNK_SIZE` in rag_pipeline.py**
- `200` → smaller chunks, sometimes incomplete answers
- `1500` → larger chunks, may include irrelevant content

**2. Change `TOP_K`**
- `1` → only the most relevant chunk, less context
- `6` → more context, but LLM may get confused

**3. Test hallucination prevention**
Ask something not in your notes — e.g. "What is list comprehension in Python?" 
The bot should say "This information is not in your notes" instead of making something up.

**4. Force rebuild vector DB**
After changing CHUNK_SIZE, delete `chroma_db/` folder and rerun, or:
```python
vectordb = setup_pipeline(force_rebuild=True)
```

---

## Tech Stack

- **LangChain** — RAG pipeline orchestration
- **ChromaDB** — local vector database
- **sentence-transformers** — free local embeddings (all-MiniLM-L6-v2)
- **Groq (llama-3.3-70b-versatile)** — LLM for answer generation
- **Streamlit** — interactive web UI

---

## Next Step — Agentic Upgrade

This pipeline can be wrapped as a `@tool` inside a LangChain agent — so the agent decides *when* to search notes vs use other tools. The `retriever.invoke()` call becomes the tool's core logic.
