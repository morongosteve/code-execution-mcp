FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000

CMD ["code-execution-mcp"]
