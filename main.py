import os
import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from sentence_transformers import SentenceTransformer
import requests
import uvicorn

from ingest_routes import get_ingest_router
from chat_routes import get_chat_router

# =========================
# Env / Config
# =========================
CHROMA_HOST = os.getenv("CHROMA_HOST", "http://localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "base_logistica")

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Ollama (opcional)
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:7869")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "mistral")

APP_PORT = int(os.getenv("APP_PORT", "9000"))

def _strip_scheme(host: str) -> str:
    return host.replace("http://", "").replace("https://", "")

# =========================
# Chroma client (simple)
# =========================
chroma_client = chromadb.HttpClient(
    host=_strip_scheme(CHROMA_HOST),
    port=CHROMA_PORT,
    settings=ChromaSettings(),
)
collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION)

# Embeddings compartidos
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# =========================
# Helpers Ollama
# =========================
def get_ollama_models(base_url: str) -> list[str]:
    """
    Devuelve lista de nombres de modelos registrados en Ollama (p.ej. ['mistral:latest', 'llama3:8b']).
    Si falla, devuelve lista vacía.
    """
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json() or {}
        models = data.get("models", []) or []
        out = []
        for m in models:
            # Preferimos 'name' si existe; si no, intentamos 'model'
            name = m.get("name") or m.get("model")
            if isinstance(name, str):
                out.append(name)
        # Orden alfabético para UX
        out = sorted(set(out))
        return out
    except Exception:
        return []

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Chroma Ingest + RAG Chat (v2)")

@app.get("/", response_class=HTMLResponse)
def home():
    # Obtenemos modelos de Ollama para el <select>
    models = get_ollama_models(OLLAMA_BASE)
    # Construimos las <option> (si no hay, mostraremos un input manual)
    options_html = ""
    if models:
        # Marcamos como selected el valor por defecto si aparece
        for m in models:
            selected = " selected" if m.startswith(OLLAMA_CHAT_MODEL) else ""
            options_html += f'<option value="{m}"{selected}>{m}</option>'

    # Render page
    return HTMLResponse(
        f"""
        <h1>Chroma Ingest + RAG Chat</h1>
        <p>Collection: <b>{CHROMA_COLLECTION}</b></p>
        <hr/>
        <h2>1) Upload Excel → Index in ChromaDB</h2>
        <form method="post" action="/ingest" enctype="multipart/form-data">
          <p><input type="file" name="files" multiple accept=".xlsx,.xls" required /></p>
          <p><button type="submit">Upload & Index</button></p>
        </form>
        <hr/>
        <h2>2) Ask the RAG Chat (with optional file uploads)</h2>
        <form method="post" action="/chat/ask" enctype="multipart/form-data">
          <p><textarea name="q" rows="4" cols="80" placeholder="Ask a question..." required></textarea></p>
          <p><label>Top K: <input type="number" name="k" value="5" min="1" max="20"></label></p>
          <p>
            <label>Modelo (Ollama):</label><br/>
            {"<select name='model'>" + options_html + "</select>" if models else
            f"<input type='text' name='model' placeholder='e.g. {OLLAMA_CHAT_MODEL}' value='{OLLAMA_CHAT_MODEL}' />"
            }
            <br/><small style="color:#666">{
                "Selecciona un modelo disponible en tu Ollama." if models
                else "No pude listar modelos de Ollama; ingresa el nombre manualmente (p. ej. 'mistral' o 'mistral:latest')."
            }</small>
          </p>
          <p>
            <label>Adjuntar archivos (opcional):
              <input type="file" name="files" multiple accept=".xlsx,.xls,.csv,.txt,.md" />
            </label>
          </p>
          <p>
            <label>
              <input type="checkbox" name="persist_uploads" value="true" />
              Persistir archivos subidos en Chroma (si no, se usan solo como contexto efímero)
            </label>
          </p>
          <p><button type="submit">Ask</button></p>
        </form>
        <hr/>
        <p style="color:#666">Embed model: <code>{EMBED_MODEL_NAME}</code> • Ollama base: <code>{OLLAMA_BASE}</code> • Default model: <code>{OLLAMA_CHAT_MODEL}</code></p>
        """
    )

# Routers
app.include_router(get_ingest_router(collection, embedder))
app.include_router(get_chat_router(collection, embedder, OLLAMA_BASE, OLLAMA_CHAT_MODEL), prefix="/chat")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
