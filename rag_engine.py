"""
RAG engine core logic (google-genai SDK).
"""

import io
import re
import uuid

import chromadb
import PyPDF2
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGEngine:
    EMBEDDING_MODEL_CANDIDATES = [
        "gemini-embedding-001",
        "gemini-embedding-2",
        "gemini-embedding-2-preview",
        "text-embedding-004",
    ]
    LLM_MODEL_CANDIDATES = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
        "gemini-1.5-flash",
    ]

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.available_models = self._discover_available_models()
        self.llm_model = self._pick_initial_model(
            self.LLM_MODEL_CANDIDATES,
            self.available_models["generatecontent"],
        )
        self.embedding_model = self._pick_initial_model(
            self.EMBEDDING_MODEL_CANDIDATES,
            self.available_models["embedcontent"],
        )
        self.chroma_client = chromadb.Client()
        self.collection_name = f"pdf_docs_{uuid.uuid4().hex[:8]}"
        self.collection = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,    # increased from 1000 — captures more complete thoughts
            chunk_overlap=400,  # increased from 200 — better continuity across chunks
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        return model_name.replace("models/", "", 1)

    @staticmethod
    def _dedupe_keep_order(values):
        deduped = []
        seen = set()
        for value in values:
            if not value or value in seen:
                continue
            deduped.append(value)
            seen.add(value)
        return deduped

    def _discover_available_models(self):
        models_by_method = {"generatecontent": [], "embedcontent": []}
        try:
            for model in self.client.models.list():
                name = self._normalize_model_name(getattr(model, "name", ""))
                methods = getattr(model, "supported_actions", None)
                if methods is None:
                    methods = getattr(model, "supported_generation_methods", None)
                methods = [m.lower() for m in (methods or [])]

                if "generatecontent" in methods:
                    models_by_method["generatecontent"].append(name)
                if "embedcontent" in methods:
                    models_by_method["embedcontent"].append(name)
        except Exception:
            # Non-fatal: calls below still have static fallback candidates.
            return models_by_method

        models_by_method["generatecontent"] = self._dedupe_keep_order(
            models_by_method["generatecontent"]
        )
        models_by_method["embedcontent"] = self._dedupe_keep_order(
            models_by_method["embedcontent"]
        )
        return models_by_method

    @staticmethod
    def _pick_initial_model(preferred, discovered):
        if not discovered:
            return preferred[0]
        for model_name in preferred:
            if model_name in discovered:
                return model_name
        return discovered[0]

    def _build_model_candidates(self, current_model, preferred, discovered):
        return self._dedupe_keep_order([current_model] + preferred + discovered)

    @staticmethod
    def _is_fallback_worthy_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        fallback_markers = [
            "not_found",
            "not found",
            "not supported",
            "permission denied",
            "resource_exhausted",
            "quota",
            "rate limit",
            "unavailable",       # 503 — model overloaded / high demand
            "503",               # HTTP 503 status code
            "service unavailable",
            "overloaded",
            "timeout",
            "connection",
        ]
        return any(marker in msg for marker in fallback_markers)

    def _embed_text(self, text: str, task_type: str):
        candidates = self._build_model_candidates(
            self.embedding_model,
            self.EMBEDDING_MODEL_CANDIDATES,
            self.available_models["embedcontent"],
        )
        last_error = None
        for model_name in candidates:
            try:
                result = self.client.models.embed_content(
                    model=model_name,
                    contents=text,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
                self.embedding_model = model_name
                return result.embeddings[0].values
            except Exception as exc:
                last_error = exc
                if self._is_fallback_worthy_error(exc):
                    continue
                raise

        raise ValueError(
            "No supported embedding model is available for this API key/project. "
            f"Tried: {', '.join(candidates)}. Last error: {last_error}"
        )

    def _generate_text(self, prompt: str) -> str:
        candidates = self._build_model_candidates(
            self.llm_model,
            self.LLM_MODEL_CANDIDATES,
            self.available_models["generatecontent"],
        )
        last_error = None
        for model_name in candidates:
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.3,
                        max_output_tokens=4096,  # increased from 1024 — allows full reasoning
                    ),
                )
                self.llm_model = model_name
                return (response.text or "").strip()
            except Exception as exc:
                last_error = exc
                if self._is_fallback_worthy_error(exc):
                    continue
                raise

        raise ValueError(
            "No supported text-generation model is available for this API key/project. "
            f"Tried: {', '.join(candidates)}. Last error: {last_error}"
        )

    def extract_pages_from_pdf(self, file_obj):
        if hasattr(file_obj, "getvalue"):
            pdf_bytes = file_obj.getvalue()
        else:
            pdf_bytes = file_obj.read()

        if not pdf_bytes:
            raise ValueError("Uploaded file is empty.")

        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            text = page.extract_text() or ""
            cleaned = text.strip()
            if cleaned:
                pages.append({"page": page_num, "text": cleaned})

        return pages, len(pdf_reader.pages)

    def chunk_pages(self, pages):
        """
        Global chunking across all pages — prevents cross-page logical units
        from being split at page boundaries. Page markers are embedded in the
        text so each chunk can be tagged with the correct page number.
        """
        # Build one continuous document with page markers
        sections = []
        for page in pages:
            sections.append(f"[PAGE {page['page']}]\n{page['text']}")
        full_text = "\n\n".join(sections)

        raw_chunks = self.text_splitter.split_text(full_text)

        chunks = []
        chunk_index = 0
        current_page = pages[0]["page"] if pages else 1

        for chunk_text in raw_chunks:
            cleaned = chunk_text.strip()
            if not cleaned:
                continue

            # Track current page from any markers present in this chunk
            page_matches = re.findall(r'\[PAGE (\d+)\]', cleaned)
            if page_matches:
                current_page = int(page_matches[-1])

            # Remove page markers from the stored chunk text
            stored_text = re.sub(r'\[PAGE \d+\]\n?', '', cleaned).strip()
            if not stored_text:
                continue

            chunks.append(
                {
                    "text": stored_text,
                    "page": current_page,
                    "chunk_id": f"chunk_{chunk_index:04d}",
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

        return chunks

    def embed_and_store(self, chunks):
        if not chunks:
            raise ValueError(
                "No extractable text was found in this PDF. "
                "If it is a scanned/image PDF, run OCR first."
            )

        if self.collection is not None:
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass

        self.collection_name = f"pdf_docs_{uuid.uuid4().hex[:8]}"
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = [
                self._embed_text(chunk["text"], "RETRIEVAL_DOCUMENT") for chunk in batch
            ]
            self.collection.add(
                embeddings=embeddings,
                documents=[c["text"] for c in batch],
                ids=[c["chunk_id"] for c in batch],
                metadatas=[{"page": c["page"], "index": c["chunk_index"]} for c in batch],
            )

    def load_pdf(self, uploaded_file):
        pages, page_count = self.extract_pages_from_pdf(uploaded_file)
        if not pages:
            raise ValueError(
                "This PDF has no extractable text. It may be scanned/image-only. "
                "Please upload a text-based PDF or OCR it first."
            )

        chunks = self.chunk_pages(pages)
        self.embed_and_store(chunks)

        total_words = sum(len(page["text"].split()) for page in pages)
        total_characters = sum(len(page["text"]) for page in pages)
        return {
            "pages": page_count,
            "chunks": len(chunks),
            "words": total_words,
            "characters": total_characters,
        }

    def retrieve_relevant_chunks(self, query, top_k=10):
        if not self.collection:
            raise ValueError("No document loaded. Please upload a PDF first.")

        query_embedding = self._embed_text(query, "RETRIEVAL_QUERY")
        collection_size = self.collection.count()
        if collection_size == 0:
            raise ValueError("No chunks are available to search.")

        # Fetch more than top_k so we can filter low-relevance chunks
        fetch_k = min(top_k * 2, collection_size)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        all_retrieved = []

        for doc, meta, distance in zip(docs, metas, distances):
            similarity = 1 - float(distance) if distance is not None else 0.0
            all_retrieved.append(
                {
                    "text": doc,
                    "page": meta.get("page", "?") if meta else "?",
                    "similarity_score": round(similarity, 3),
                }
            )

        # Filter chunks below relevance threshold; fallback to top-2 if all filtered
        MIN_RELEVANCE = 0.30
        filtered = [c for c in all_retrieved if c["similarity_score"] >= MIN_RELEVANCE]
        if not filtered:
            filtered = all_retrieved[:2]

        return filtered[:top_k]

    def generate_answer(self, query, context_chunks, chat_history=None):
        if not context_chunks:
            return "I could not find this in the document."

        context = "\n\n---\n\n".join(
            [
                f"[Page {c['page']} | Relevance: {c['similarity_score']}]\n{c['text']}"
                for c in context_chunks
            ]
        )

        # Build conversation history block for multi-turn awareness
        history_text = ""
        if chat_history:
            recent = [m for m in chat_history if m.get("role") in ("user", "assistant")][-6:]
            if recent:
                lines = []
                for m in recent:
                    role = "User" if m["role"] == "user" else "Assistant"
                    lines.append(f"{role}: {m.get('content', '').strip()}")
                history_text = "CONVERSATION HISTORY (for context):\n" + "\n".join(lines) + "\n\n"

        prompt = f"""You are an expert document analyst. Answer the user's question using the document context provided.

INSTRUCTIONS:
- Read ALL context chunks carefully before answering
- Synthesize and connect information across multiple chunks when needed
- Use logical reasoning and inference — do not just copy text verbatim
- For complex questions, structure your answer clearly (use bullet points or sections)
- Cite page numbers (e.g. "Page 3") for specific facts
- If part of the question cannot be answered from the context, explicitly say so for that part only
- Use the conversation history to understand follow-up questions and references

{history_text}DOCUMENT CONTEXT:
{context}

USER QUESTION: {query}

Provide a thorough, well-reasoned answer:"""
        return self._generate_text(prompt)

    def query(self, question, chat_history=None):
        cleaned_question = (question or "").strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")
        relevant_chunks = self.retrieve_relevant_chunks(cleaned_question, top_k=10)
        answer = self.generate_answer(cleaned_question, relevant_chunks, chat_history=chat_history)
        return {"answer": answer, "sources": relevant_chunks}
