# Run containers
docker compose up -d
# Run app
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 9000 --reload