# KnowledgeHub

![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)
![LLM](https://img.shields.io/badge/LLM-OpenAI_Compatible-7C3AED)
[![CI](https://github.com/kq409/researchpilot/actions/workflows/ci.yml/badge.svg)](https://github.com/kq409/researchpilot/actions/workflows/ci.yml)

Knowledge base for capturing ideas, organizing papers, and asking questions against your own library.

Voice notes go through local Whisper transcription and LLM cleanup, then become structured research notes. Papers live in a searchable library. Chat retrieves over what you have stored, and can search the web for field-wide or latest-progress questions when configured.

**Features:**

- Voice notes: browser recording or audio upload, Whisper speech-to-text, optional LLM cleanup
- Structured research notes with review status (generated → draft → reviewed → accepted)
- Paper library with metadata, tags, and GROBID parsing
- Chat: retrieve from notes and papers, with citations; optional web search for SOTA / ArXiv / outside the library
- Compare: side-by-side comparison across selected papers
- Eval: retrieval and agent suites with a transcript viewer
- MCP: read-only library tools for Cursor
- OpenAI-compatible LLM API (Ollama, LM Studio, OpenAI, or any compatible endpoint)

---

## Quick Start

### Dev Container (Recommended)

**This project is devcontainer-first.**

#### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [VS Code](https://code.visualstudio.com/)
- [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)

#### 2. Open in Dev Container

- Click **"Reopen in Container"** in VS Code
- Or: `Cmd/Ctrl+Shift+P` → **"Dev Containers: Reopen in Container"**
- Wait ~5-10 minutes for initial build and model download

VS Code automatically:

1. Builds and starts the app, Ollama, PostgreSQL, and GROBID containers
2. Installs Python and Node.js dependencies
3. Downloads the Ollama models
4. Creates `backend/.env` with working defaults

Skip to [Running the App](#running-the-app).

---

### GitHub Codespaces

GitHub Codespaces can run this project's devcontainer in the cloud.

1. Open this repository on GitHub → **"Code"** → **"Codespaces"** → **"Create codespace"**
2. The devcontainer asks for at least **4 cores**; more CPU and RAM is better
3. Wait ~5-10 minutes for initial setup

Ports 3000, 8000, and 11434 are forwarded automatically. Use the port 3000 link or the **Ports** tab for the frontend.

If you need true `localhost` access:

1. Install the [GitHub Codespaces extension](https://marketplace.visualstudio.com/items?itemName=GitHub.codespaces) in VS Code Desktop
2. Connect to the running Codespace
3. Ports forward to your machine's `localhost`

Stop the Codespace when idle at [github.com/codespaces](https://github.com/codespaces). Any platform that supports devcontainers (Gitpod, DevPod, etc.) can also use `.devcontainer`.

---

### Manual Installation

The devcontainer is the supported setup. If you install manually, you need:

- Python 3.12+, Node.js 24+, [uv](https://docs.astral.sh/uv/), PostgreSQL with pgvector, and an LLM server ([Ollama](https://ollama.com/) or [LM Studio](https://lmstudio.ai/))
- Copy `backend/.env.example` to `backend/.env` and configure
- Install dependencies with `uv sync` (backend) and `npm install` (frontend)
- Start your LLM server and pull models: `ollama pull gemma3:4b` and `ollama pull nomic-embed-text`

---

## Running the App

Open **two terminals** and run:

**Terminal 1 - Backend:**

```bash
cd backend
uv sync && uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000 --timeout-keep-alive 600
```

`uv sync` refreshes dependencies. `--timeout-keep-alive 600` allows long audio and paper processing.

**Terminal 2 - Frontend:**

```bash
cd frontend
npm install && npm run dev
```

**Browser:** Open `http://localhost:3000`

---

## Configuration

KnowledgeHub talks to any OpenAI-compatible LLM provider:

- **Ollama** (default in the devcontainer)
- **LM Studio**
- **OpenAI API**
- Any other OpenAI-compatible API

The devcontainer creates `backend/.env` with Ollama defaults. To switch providers, edit:

- `LLM_BASE_URL` — API endpoint
- `LLM_API_KEY` — API key (never commit real keys)
- `LLM_MODEL` — Model name
- `LLM_REASONING_EFFORT` — global default `none` / `low` / `medium` / `high`
- Optional per-path overrides: `ASK_REASONING_EFFORT`, `CHAT_REASONING_EFFORT`,
  `COMPARE_REASONING_EFFORT`, `JUDGE_REASONING_EFFORT` (Judge defaults to `none`)
- `ASK_MAX_TOKENS` / `CHAT_MAX_TOKENS` — generation ceilings (default **4096**)
- `COMPARE_MAX_TOKENS` / `AGENT_COMPARE_MAX_TOKENS` — Compare ceilings (raise if
  high reasoning effort truncates JSON)
- `JUDGE_MODEL` — cheaper reviewer when the main model is a large reasoning model

### DeepSeek (chat) + Ollama (embeddings)

Keep chat and embeddings on separate hosts — DeepSeek does not serve
`nomic-embed-text`. Example `backend/.env` block:

```env
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-your-key
LLM_MODEL=deepseek-v4-flash
LLM_REASONING_EFFORT=high
ASK_REASONING_EFFORT=none
CHAT_REASONING_EFFORT=low
JUDGE_REASONING_EFFORT=none
JUDGE_MODEL=deepseek-v4-flash
ASK_MAX_TOKENS=4096
CHAT_MAX_TOKENS=4096

EMBEDDING_BASE_URL=http://ollama:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL=nomic-embed-text
```

If Ask/Chat return HTTP 404 mentioning an embedding model, the chat host is
being used for embeddings — fix `EMBEDDING_*`, do not point them at DeepSeek.

Chat `web_search` (field-wide / SOTA / ArXiv questions) uses the DeepSeek
Responses API, independent of `LLM_BASE_URL`:

```env
WEB_SEARCH_BASE_URL=https://api.deepseek.com
WEB_SEARCH_API_KEY=sk-your-key
WEB_SEARCH_MODEL=deepseek-v4-flash
```

Dummy values (`ollama`, `changeme`, empty) count as unset. Without a real key,
Chat says web search is not configured and stays in the library.

---

## Tests and CI

Push and pull requests to `main` run [GitHub Actions](.github/workflows/ci.yml). Two jobs run in parallel:

- **Backend:** pgvector Postgres, Alembic migrations, Ruff, Black, pytest
- **Frontend:** Node 24, `npm ci`, type-check, ESLint, production build

LLM, GROBID, and Whisper are mocked in tests. You do not need Ollama for CI.

**Local equivalents** (Postgres with pgvector must be running; the devcontainer already provides it):

```bash
# backend
cd backend
uv sync --extra dev
uv run alembic upgrade head
uv run ruff check .
uv run black --check .
uv run pytest --tb=short

# frontend
cd frontend
npm ci
npm run type-check
npm run lint
npm run build
```

Set `DATABASE_URL` (see `backend/.env.example`). Without it, API tests that need Postgres skip and the suite can look green while skipping most coverage.

This CI does not deploy the app.

### Evaluation

Ask (retrieval + abstain + citations) and Chat (tool/outcome) suites live in `backend/eval/suites/`. They seed a tagged synthetic ZXQ* library, run the production `AskService` / `ResearchAgent`, and write transcripts to `backend/eval/runs/`.

```bash
cd backend
uv run python -m eval.harness --suite ask
uv run python -m eval.harness --suite chat --llm
```

Without `--llm`, Ask uses keyword embeddings and a stub completer so retrieval graders run offline. Open the Eval panel in the app header to browse the latest run and transcripts. `POST /api/eval/run?suite=ask` triggers the same Ask suite.

### MCP (read-only library tools)

The Chat subagent tools (`search_library`, `list_papers`, `list_notes`, `read_paper`, `read_note`, `list_skills`, `load_skill`, `memory_search`) are also an MCP server. Postgres and the embedding host must already be running. Cursor `mcp.json`:

```json
{
  "mcpServers": {
    "knowledgehub": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server"],
      "cwd": "/absolute/path/to/researchpilot/backend"
    }
  }
}
```

Write tools (link/unlink, memory_write, compare) are not exposed.

The chat agent can also *consume* external MCP tools. Copy
`backend/mcp_servers.example.json` to `backend/mcp_servers.json` (or set
`MCP_SERVERS`) with an `allow` list of remote tool names. Names show up as
`mcp__{server}__{tool}`. A server's own `readOnlyHint` is not authorisation;
anything not on `allow` needs in-chat approval.

Write tools in Chat (`memory_write`, `memory_delete`, `link_note`,
`unlink_note`, `connect_note`) pause for a researcher click unless
`ASK_APPROVAL_MODE=off`. The wait is process-local: the replica that started
the turn must receive `POST /api/chat/approve`.

Hybrid lexical search uses stored `tsvector` columns. Optional FlashRank rerank: set `RERANK_ENABLED=true` (downloads a small ONNX model on first use). Ask/Chat traces always append JSONL under `data/traces/`; set `OPIK_API_KEY` to also send them to Comet Opik.

---

## Troubleshooting

**Container won't start or is very slow:**

This stack runs an LLM on CPU unless a GPU is available. Give Docker enough resources:

1. Open **Docker Desktop** → **Settings** → **Resources**
2. Set **CPUs** as high as you can (8+ cores recommended)
3. Set **Memory** to at least 16GB
4. Click **Apply & Restart**

Expected specs: modern machine with 8+ CPU cores and 16GB RAM.

**Microphone not working:**

- Use Chrome or Firefox (Safari may have issues)
- Check browser permissions: Settings → Privacy → Microphone

**Backend fails to start:**

- Check Whisper model downloads: `~/.cache/huggingface/`
- Ensure enough disk space (models are ~150MB)

**LLM errors:**

- Make sure Ollama is running (it auto-starts with the devcontainer)
- Confirm models downloaded during setup
- Voice transcription still works without the LLM (raw Whisper only)

**LLM is slow:**

- See Docker resource settings above
- Switch `LLM_MODEL` in `backend/.env` (smaller models are faster, weaker at cleanup and extraction)
- Use a cloud API such as OpenAI for faster, higher-quality responses

**Cannot access localhost:3000 or localhost:8000 from the host:**

- **Docker Desktop:** **Settings** → **Resources** → **Network**
- Enable **"Use host networking"** (may require a restart)
- Restart the frontend and backend servers

**Port already in use:**

- Backend: change port with `--port 8001`
- Frontend: edit `vite.config.ts`, change `port: 3000`
