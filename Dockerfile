FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

ENV API_BASE_URL=""
ENV MODEL_NAME="claude-haiku-4-5-20251001"
ENV HF_TOKEN=""

CMD ["python", "server.py"]
