import uuid
from io import BytesIO
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import HTMLResponse

def get_ingest_router(collection, embedder):
    router = APIRouter()

    def dataframe_to_text(df: pd.DataFrame, nombre: str) -> str:
        texto = f"Información del archivo {nombre}:\n"
        for _, fila in df.iterrows():
            fila_texto = ", ".join([f"{col}: {fila[col]}" for col in df.columns])
            texto += f"- {fila_texto}\n"
        return texto

    def xlsx_to_text(file_bytes: bytes, filename: str) -> str:
        df = pd.read_excel(BytesIO(file_bytes))
        return dataframe_to_text(df, Path(filename).stem)

    @router.post("/ingest", response_class=HTMLResponse)
    async def ingest(files: List[UploadFile] = File(...)):
        added = 0
        msgs = []
        for f in files:
            data = await f.read()
            try:
                text = xlsx_to_text(data, f.filename)
                vec = embedder.encode([text])[0].tolist()
                doc_id = f"doc_{uuid.uuid4()}"
                collection.add(
                    ids=[doc_id],
                    documents=[text],
                    embeddings=[vec],
                    metadatas=[{"source": f.filename}],
                )
                added += 1
                msgs.append(f"✅ Indexed: {f.filename} → {doc_id[:8]}")
            except Exception as e:
                msgs.append(f"❌ {f.filename}: {e}")
        return HTMLResponse(
            "<br>".join(
                [
                    f"<b>Collection:</b> {collection.name if hasattr(collection, 'name') else ''}",
                    f"<b>Added:</b> {added}",
                    *msgs,
                    '<p><a href="/">Back</a></p>',
                ]
            )
        )

    return router
