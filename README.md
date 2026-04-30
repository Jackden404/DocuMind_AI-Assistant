# DocuMind AI Assistant - PDF RAG Project

DocuMind is a local-first Streamlit application that lets you upload a PDF and chat with it using Retrieval-Augmented Generation (RAG).

It combines:
- PDF parsing (`PyPDF2`)
- intelligent chunking (`langchain-text-splitters`)
- vector storage and retrieval (`ChromaDB`)
- embeddings and answer generation (`Google Gemini API`)
- a dark-mode chat UI (`Streamlit`)

---

## 1. Project Goal

The goal is to make long documents searchable and conversational:
- upload a PDF
- process and index its text
- ask natural language questions
- get grounded answers with source chunks and page references

---

## 2. Key Features

- Modern dark chat interface (ChatGPT/Claude-inspired layout)
- Top upload section with:
  - drag and drop uploader
  - file preview
  - centered "Process Document" button
- Middle scrollable chat area:
  - user messages on right
  - assistant messages on left
- Bottom fixed chat input:
  - placeholder: "Ask your questions about the document..."
  - disabled until processing is complete
- Sidebar controls:
  - API key input
  - connection status
  - indexed document stats
- Source-aware answers:
  - retrieved chunk snippets
  - page numbers
  - similarity relevance values
- Model resiliency:
  - auto-discovery of available Gemini models
  - fallback across multiple embedding/generation model candidates

---

## 3. Architecture

```text
PDF Upload
   ->
Text Extraction (PyPDF2)
   ->
Page-wise Chunking (RecursiveCharacterTextSplitter)
   ->
Embeddings (Gemini embedContent API)
   ->
Vector Store (ChromaDB cosine index)
   ->
Query Embedding + Top-K Retrieval
   ->
Prompt Construction with Retrieved Context
   ->
Answer Generation (Gemini generateContent API)
   ->
Answer + Source Chunks in UI
```

---

## 4. Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit | Web app and chat interface |
| Env | python-dotenv | Load `.env` variables |
| PDF parsing | PyPDF2 | Extract text from PDFs |
| Text split | langchain-text-splitters | Chunking with overlap |
| Vector DB | ChromaDB | In-memory vector search |
| Embeddings + LLM | google-genai | Gemini embeddings and generation |

---

## 5. Project Structure

```text
PDF_Assistant/
|-- app.py             # Streamlit UI and interaction flow
|-- rag_engine.py      # RAG pipeline (extract, chunk, embed, retrieve, answer)
|-- requirements.txt   # Python dependencies
|-- .env               # Local environment variables (not for public sharing)
|-- README.md          # Project documentation
```

---

## 6. Setup Instructions

## 6.1 Prerequisites

- Python 3.10 or newer
- pip
- Gemini API key (from Google AI Studio)

Get API key:
- https://aistudio.google.com/app/apikey

## 6.2 Install Dependencies

From project root:

```bash
pip install -r requirements.txt
```

## 6.3 Configure Environment

Create/update `.env`:

```env
GOOGLE_API_KEY=your_api_key_here
```

## 6.4 Run the App

```bash
streamlit run app.py
```

Default local URL:
- http://localhost:8501

---

## 7. How to Use

1. Enter Gemini API key in the sidebar.
2. Upload a PDF in the top upload section.
3. Click **Process Document**.
4. Wait for success banner ("Document processed. Chat input is now enabled.").
5. Ask questions in the bottom chat input.
6. Review source chunks under assistant responses.

---

## 8. Core RAG Logic (Current Implementation)

Important implementation points in `rag_engine.py`:

- **Model discovery and fallback**
  - discovers available models via `client.models.list()`
  - dynamically picks compatible models for `generateContent` and `embedContent`
  - retries with fallback candidates when errors indicate model/quota/support problems

- **Page-wise chunking**
  - extracts text per page
  - splits each page into overlapping chunks
  - stores page metadata for better citation quality

- **Indexing and retrieval**
  - stores embeddings in ChromaDB (cosine metric)
  - retrieves top-k chunks for each query
  - includes similarity score in returned sources

- **Grounded answer generation**
  - prompt instructs assistant to answer from provided context only
  - asks to cite page numbers
  - returns fallback message when answer is not found

---

## 9. Privacy and Data Handling

This app is local-first for UI and document workflow, but it is **not fully offline**:

- Local:
  - app runtime
  - file upload handling
  - chunking and vector storage (in-process ChromaDB)
- Cloud/API:
  - chunk text and user queries are sent to Gemini API for embeddings and generation

Accurate privacy statement:
- "The application runs locally and gives better control than public upload tools, but model inference uses cloud API calls."

---

## 10. Known Limitations

- Scanned/image-only PDFs without OCR will fail (no extractable text).
- Very large PDFs may take longer and consume more memory.
- API usage depends on Gemini quota/rate limits.
- ChromaDB is in-memory for current runtime session.

---

## 11. Troubleshooting

## A) `404 model not found`

The app already includes dynamic model fallback. If this still occurs:
- update dependencies
- verify API key has Gemini API access

## B) `429 RESOURCE_EXHAUSTED`

- free-tier quota is exhausted
- wait and retry
- or use a billing-enabled project/key

## C) `No extractable text was found`

- PDF is likely scanned/image-based
- run OCR first, then upload OCR text PDF

## D) `ModuleNotFoundError`

Reinstall dependencies:

```bash
pip install -r requirements.txt
```

---

## 12. Future Improvements

- persistent on-disk vector store
- multi-document workspace
- OCR pipeline integration for scanned PDFs
- chat history export
- streaming token-by-token responses
- authentication and role-based access for team use

---

## 13. Quick Viva/Interview Summary

- Problem solved: fast Q and A over long PDF documents
- Approach: RAG (retrieve relevant chunks before generation)
- Why reliable: grounded answers + source chunk visibility
- Why practical: real-world workflow with modern LLM stack and usable UI

