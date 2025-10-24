# DIS RAG

## Run containers

docker compose up -d

## Run app

- python3 -m venv .env
- source .venv/bin/activate
- pip install -r requirements.txt
- uvicorn main:app --host 0.0.0.0 --port 9000 --reload

## User query

Genera la asignación del total de pedidos a camiones disponibles.
