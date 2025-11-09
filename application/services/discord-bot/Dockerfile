FROM python:3.11-slim

# System-Pakete
RUN apt-get update && apt-get install -y \
    gcc curl && \
    rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis
WORKDIR /app

# Python-Pakete
RUN pip install --no-cache-dir \
    discord.py==2.3.2 \
    aiohttp==3.9.1 \
    pydantic==2.5.0 \
    python-dotenv==1.0.0 \
    requests==2.31.0

# Code kopieren
COPY src/ ./src/

# Healthcheck
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import discord; print('OK')" || exit 1

# Start
CMD ["python", "src/bot.py"]