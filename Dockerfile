FROM python:3.12-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://hf.co/cli/install.sh | bash && \
    ln -sf /root/.local/bin/hf /usr/local/bin/hf

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -e ".[all]"

EXPOSE 8000

CMD ["code-execution-mcp"]
