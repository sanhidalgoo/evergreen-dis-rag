from typing import List
import requests
from fastapi import APIRouter, Form
from fastapi.responses import HTMLResponse

def get_chat_router(collection, embedder, ollama_base: str, ollama_model: str):
    router = APIRouter()

    def answer_with_ollama(contexts: List[str], question: str) -> str:
        try:
            url = f"{ollama_base}/api/chat"
            sys = (
                "You are a helpful assistant. Answer only using the provided context. "
                "If the context is insufficient, say you don't know."
            )
            user = f"Context:\n{'\\n\\n---\\n\\n'.join(contexts)}\n\nQuestion: {question}\nAnswer:"
            r = requests.post(
                url,
                json={
                    "model": ollama_model,
                    "messages": [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.5},
                },
                timeout=120,
            )
            r.raise_for_status()
            return (r.json().get("message") or {}).get("content", "").strip()
        except Exception:
            return "(No LLM available) Top matches:\n\n" + "\n\n---\n\n".join(contexts)

    @router.post("/ask", response_class=HTMLResponse)
    async def ask(q: str = Form(...), k: int = Form(5)):
        qvec = embedder.encode([q])[0].tolist()
        res = collection.query(
            query_embeddings=[qvec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0] if res and res.get("documents") else []
        if not docs:
            return HTMLResponse("<p><b>No matches found</b>. Did you ingest documents?</p><p><a href='/'>Back</a></p>")

        answer = answer_with_ollama(docs, q)
        html = f"""
        <h3>Answer</h3>
        <div style="white-space:pre-wrap;border:1px solid #ddd;padding:10px;border-radius:8px;">{answer}</div>
        <h4>Top {k} context chunks</h4>
        <ol>
        {''.join([f"<li><details><summary>Chunk {i+1}</summary><pre style='white-space:pre-wrap'>{docs[i]}</pre></details></li>" for i in range(len(docs))])}
        </ol>
        <p><a href="/">Back</a></p>
        """
        return HTMLResponse(html)

    return router
