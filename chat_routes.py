import uuid
from io import BytesIO
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
from fastapi import APIRouter, Form, File, UploadFile
from fastapi.responses import HTMLResponse

def get_chat_router(collection, embedder, ollama_base: str, ollama_model: str):
    router = APIRouter()

    # ---------- Helpers para lectura de archivos ----------
    def dataframe_to_text(df: pd.DataFrame, nombre: str) -> str:
        texto = f"Información del archivo {nombre}:\n"
        for _, fila in df.iterrows():
            fila_texto = ", ".join([f"{col}: {fila[col]}" for col in df.columns])
            texto += f"- {fila_texto}\n"
        return texto

    def excel_to_text(file_bytes: bytes, filename: str) -> str:
        df = pd.read_excel(BytesIO(file_bytes))
        return dataframe_to_text(df, Path(filename).stem)

    def csv_to_text(file_bytes: bytes, filename: str) -> str:
        # Intento con UTF-8 y fallback a Latin-1; pandas deduce separador si es posible
        try:
            df = pd.read_csv(BytesIO(file_bytes))
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(file_bytes), encoding="latin-1")
        return dataframe_to_text(df, Path(filename).stem)

    def text_like_to_text(file_bytes: bytes) -> str:
        try:
            return file_bytes.decode("utf-8", errors="replace")
        except Exception:
            return ""

    def answer_with_ollama(contexts: List[str], question: str) -> str:
        try:
            url = f"{ollama_base}/api/chat"
            # sys = (
            #     "Eres un asistente muy útil para mapear pedidos a camiones. Responde solo utilizando el contexto proporcionado"
            #     "Serás experto generando la lista de pedidos que tendrá cada camión"
            # )
            sys = """# Sistema de Asignación de Despacho de Pedidos

            Eres un asistente experto en logística y optimización de rutas, especializado en la generación de archivos de despacho de pedidos. Tienes acceso a una base de conocimiento que contiene información sobre SLAs, políticas de entrega, restricciones de productos y reglas de negocio.

            La información de camiones disponibles y pedidos pendientes está disponible en los archivos cargados en el sistema. La fecha de despacho es HOY.

            ---

            ## INSTRUCCIONES DE ASIGNACIÓN:

            Genera una asignación óptima de pedidos a camiones siguiendo estos criterios en orden de prioridad:

            ### 1. CUMPLIMIENTO DE SLA (Prioridad Crítica)
            - Consulta los SLA de cada plan de cliente en la base de conocimiento
            - Los pedidos con SLA más restrictivo tienen prioridad absoluta
            - Identifica pedidos en riesgo de incumplimiento y márcalos

            ### 2. RESTRICCIONES DE CARROCERÍA (Obligatorio)
            - *Refrigerado*: Solo puede ir en camiones con carrocería refrigerada
            - *Seco*: Puede ir en carrocería seca o mixta
            - *Mixto*: Puede combinar productos si el camión lo permite
            - NUNCA asignes un pedido a un camión incompatible

            ### 3. CAPACIDAD FÍSICA (Obligatorio)
            - Verifica que el peso total asignado ≤ capacidad de peso del camión
            - Verifica que el volumen total asignado ≤ capacidad volumétrica del camión
            - Calcula el % de utilización de cada camión

            ### 4. OPTIMIZACIÓN DE RUTAS
            - Agrupa pedidos con destinos cercanos en el mismo camión
            - Prioriza rutas que maximicen la eficiencia del viaje
            - Considera la secuencia lógica de entrega

            ---

            ## FORMATO DE SALIDA REQUERIDO:

            Genera la asignación en el siguiente formato estructurado:

            DESPACHO - [FECHA DE HOY]
            =====================================

            CAMIÓN: [ID_CAMION] - [PLACA]
            Tipo Carrocería: [TIPO]
            Capacidad: [PESO_MAX]kg / [VOLUMEN_MAX]m³

            Pedidos Asignados:
            1. Pedido #[ID] - Cliente: [NOMBRE] (Plan: [PLAN])
            - Destino: [CIUDAD/ZONA]
            - Peso: [X]kg | Volumen: [Y]m³
            - Tipo: [REFRIGERADO/SECO/MIXTO]
            - SLA: [DÍAS] días

            2. [...]

            Totales:
            - Peso Total: [X]kg ([%]% de capacidad)
            - Volumen Total: [Y]m³ ([%]% de capacidad)
            - Pedidos: [N]
            - Ruta Sugerida: [DESCRIPCIÓN]

            Notas: [ALERTAS O CONSIDERACIONES ESPECIALES]

            -------------------------------------

            [REPETIR PARA CADA CAMIÓN]

            =====================================

            RESUMEN GENERAL:
            - Total Camiones Utilizados: [N]
            - Total Pedidos Asignados: [N]
            - Pedidos No Asignados: [N] (listar con razón)
            - Utilización Promedio Flota: [%]%

            ALERTAS CRÍTICAS:
            [Lista de pedidos en riesgo de SLA o advertencias importantes]


            ---

            ## VALIDACIONES OBLIGATORIAS:

            Antes de generar la salida, verifica:
            - ✓ Ningún pedido excede capacidad del camión asignado
            - ✓ Todos los pedidos cumplen restricción de carrocería
            - ✓ Pedidos urgentes están asignados prioritariamente
            - ✓ No hay conflictos de incompatibilidad (ej: productos que no pueden ir juntos)

            ## REGLAS ADICIONALES:

            - Si un pedido NO puede ser asignado, explica claramente el motivo
            - Si hay conflicto entre criterios, prioriza: SLA > Carrocería > Capacidad > Optimización
            - Sugiere acciones cuando la capacidad disponible es insuficiente
            - Marca con ⚠️ cualquier asignación que esté al límite de capacidad (>95%)
            - Consulta la base de conocimiento para cualquier regla específica de cliente o producto
            - Si la información es insuficiente o no está disponible en la base de conocimiento, indícalo claramente
            ---

            ## RESPUESTA:
            """
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

    # ---------- Endpoint principal ----------
    @router.post("/ask", response_class=HTMLResponse)
    async def ask(
        q: str = Form(...),
        k: int = Form(5),
        files: Optional[List[UploadFile]] = File(None),
        persist_uploads: Optional[str] = Form(None),  # "true" si el checkbox viene marcado
    ):
        # 1) Recuperación con la colección existente
        qvec = embedder.encode([q])[0].tolist()
        res = collection.query(
            query_embeddings=[qvec],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res.get("documents", [[]])[0] if res and res.get("documents") else []

        # 2) Procesar archivos subidos (si hay)
        uploaded_texts: List[str] = []
        persisted_count = 0

        if files:
            for f in files:
                try:
                    file_bytes = await f.read()
                    name = (f.filename or "").lower()

                    if name.endswith(".xlsx") or name.endswith(".xls"):
                        text = excel_to_text(file_bytes, f.filename)
                    elif name.endswith(".csv"):
                        text = csv_to_text(file_bytes, f.filename)
                    elif name.endswith(".txt") or name.endswith(".md"):
                        text = text_like_to_text(file_bytes)
                    else:
                        # Tipo no soportado: lo ignoramos con una marca
                        uploaded_texts.append(f"(Ignorado {f.filename}: tipo no soportado)")
                        continue

                    # Añadimos el texto a contexto efímero
                    uploaded_texts.append(f"(Subido) {f.filename}:\n{text}")

                    # Si se pidió persistir, lo indexamos en Chroma
                    if persist_uploads == "true":
                        vec = embedder.encode([text])[0].tolist()
                        doc_id = f"chatdoc_{uuid.uuid4()}"
                        collection.add(
                            ids=[doc_id],
                            documents=[text],
                            embeddings=[vec],
                            metadatas=[{"source": f"(chat_upload) {f.filename}"}],
                        )
                        persisted_count += 1

                except Exception as e:
                    uploaded_texts.append(f"(Error leyendo {f.filename}: {e})")

        # 3) Construir el contexto final (archivos subidos primero, luego los chunks de la colección)
        contexts = uploaded_texts + docs

        if not contexts:
            return HTMLResponse(
                "<p><b>No matches found</b>. "
                "No hay documentos en la colección y no se subieron archivos.</p>"
                "<p><a href='/'>Back</a></p>"
            )

        # 4) LLM (o fallback)
        answer = answer_with_ollama(contexts, q)

        # 5) Render de respuesta + fuentes
        def li_block(title: str, body: str) -> str:
            return (
                "<li><details><summary>"
                + title
                + "</summary><pre style='white-space:pre-wrap'>"
                + body
                + "</pre></details></li>"
            )

        uploaded_list = "".join(
            [li_block(f"(Upload) {i+1}", txt) for i, txt in enumerate(uploaded_texts)]
        )
        retr_list = "".join(
            [
                li_block(f"(DB) Chunk {i+1}", docs[i])
                for i in range(len(docs))
            ]
        )

        persist_note = (
            f"<p><i>Persistidos en Chroma: {persisted_count}</i></p>"
            if persist_uploads == "true"
            else "<p><i>Los archivos subidos se usaron solo como contexto efímero.</i></p>"
            if files
            else ""
        )

        html = f"""
        <h3>Answer</h3>
        <div style="white-space:pre-wrap;border:1px solid #ddd;padding:10px;border-radius:8px;">{answer}</div>
        {persist_note}
        <h4>Contexto de esta respuesta</h4>
        <ol>
          {uploaded_list}
          {retr_list}
        </ol>
        <p><a href="/">Back</a></p>
        """
        return HTMLResponse(html)

    return router
