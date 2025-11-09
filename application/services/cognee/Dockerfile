FROM python:3.11-slim

# System-Pakete
RUN apt-get update && apt-get install -y \
    curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Python-Pakete
RUN pip install --no-cache-dir \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    asyncpg==0.29.0 \
    sentence-transformers==2.2.2 \
    numpy==1.24.3 \
    networkx==3.2.1 \
    python-dotenv==1.0.0 \
    openai==1.3.5

# Code kopieren
COPY src/ ./src/
COPY init/ ./init/

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]