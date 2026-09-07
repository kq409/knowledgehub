# ResearchPilot

Personal research assistant for capturing ideas, organizing papers, and asking questions against your own library.

Voice notes go through local Whisper transcription and LLM cleanup, then become structured research notes. Papers live in a searchable library. Ask and Compare run retrieval over what you have stored.

**Features:**

- Voice notes: browser recording or audio upload, Whisper speech-to-text, optional LLM cleanup
- Structured research notes with review status (generated → draft → reviewed → accepted)
- Paper library with metadata, tags, and GROBID parsing
- Ask: retrieve from notes and papers
- Compare: side-by-side comparison across selected papers
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

ResearchPilot talks to any OpenAI-compatible LLM provider:

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
