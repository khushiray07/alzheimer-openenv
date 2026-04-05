# ── Stage 1: Build React frontend ─────────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python FastAPI server ────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Copy the compiled React build into ./static
COPY --from=frontend-build /frontend/dist ./static

EXPOSE 7860

ENV API_BASE_URL="https://api.anthropic.com"
ENV MODEL_NAME="claude-haiku-4-5-20251001"
# HF_TOKEN has no default — must be injected as a secret

CMD ["python", "server.py"]
