# 火电两票协同智能审查与设备运维多 Agent 系统 — 后端镜像
FROM python:3.11-slim

WORKDIR /app

# 系统依赖（jieba/向量/OCR 等运行所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY knowledge_base/ knowledge_base/
COPY models/ models/
COPY eval/ eval/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
