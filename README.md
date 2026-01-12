# MultiAgentEnterpriseRAG

Multi-agent RAG demo for interacting with company information: ingest local documents into Qdrant, then chat via a LangGraph-orchestrated workflow that retrieves, reasons, and returns structured citations, with multi-turn memory in Redis.

For deeper design details (agents, state, data flow, decisions), see `ARCHITECTURE.md`.

## Features

- Document ingestion endpoint (`POST /ingest`) that indexes local file paths into the vector store. 
- Multi-agent chat endpoint (`POST /chat`) backed by a LangGraph workflow (supervisor → planner → retrieval → quality gate → reasoning → citation → memory). 
- Hybrid retrieval (dense vector search + lexical text match) with lightweight deterministic reranking. 
- Multi-turn memory using Redis (recent window + rolling summary with TTL).   
- Minimal browser UI (`ui.html`) for ingestion + chat, including citation rendering. 

## Quickstart (Docker Compose)

Prerequisits: Docker + Docker Compose.

1) Clone this repository on your machine: 
```bash
git clone https://github.com/pchainieux/MultiAgentEnterpriseRAG.git
```

2) Create a `.env` file, copy it from the following template:
```text
# ---------- Core app ----------
APP_ENV=local

# ---------- Qdrant ----------
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=documents
QDRANT_VECTOR_DIM=1024

# ---------- Redis ----------
REDIS_URL=redis://redis:6379/0

# ---------- LLM / embeddings ----------
LLM_PROVIDER=ollama  # choose between openai and ollama

OPENAI_API_KEY= add-your-own-api-key
OPENAI_MODEL_NAME=gpt-4.1-mini
EMBEDDING_MODEL_NAME=BAAI/bge-large-en-v1.5

OLLAMA_BASE_URL=https://ollama.com/api
OLLAMA_API_KEY=add-your-own-api-key
OLLAMA_MODEL=gpt-oss:120b

# ---------- Logging / observability ----------
LOG_LEVEL=INFO
```
and set at least your own LLM API keys. 

3) Start the stack:
```bash
docker compose up --build
```
And visit `http://localhost:8000/`to use the built in ingestion + chat UI.

## Repository layout
- src/app/: FastAPI service (routers, deps, config, logging, schemas). 
- src/graph/: LangGraph orchestration (shared state, workflow wiring, and agent nodes). 
- src/rag/: RAG primitives (Qdrant client, hybrid retrieval, LLM adapters/prompts, Redis memory helpers). 

## API usage
### Ingesting 
The UI expects container-local file paths to ingest them (for example: ./data/docs/document.txt)

```bash 
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "paths": ["./data/docs/document.pdf", "./data/docs/document.txt"]
  }'
```

### Chatting
Runs the LangGraph workflow using sessionid for multi-turn continuity and returns an answer plus structured citations.

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "sessionid": "demo-session-1",
    "messages": [
      {"role": "user", "content": "example_question"}
    ]
  }'

```

## Tests
The repo includes pytest dependencies, run tests locally from the repo root once you have a Python environment configured: 

```bash
PYTHONPATH=. pytest
```

## License
N/A
