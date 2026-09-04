#!/bin/bash
set -e

echo "🚀 Setting up ResearchPilot development environment..."

# Ensure cache directories exist with proper permissions
mkdir -p "$HOME/.cache/uv" "$HOME/.cache/huggingface"

# Wait for Ollama service to be ready
echo "⏳ Waiting for Ollama service..."
for i in {1..30}; do
    if curl -s http://ollama:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama service is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Warning: Ollama service not responding (you can start it later)"
    fi
    sleep 1
done

# Create .env from example if it doesn't exist
if [ ! -f backend/.env ]; then
    echo "📝 Creating backend/.env from backend/.env.example..."
    cp backend/.env.example backend/.env
    echo "✅ backend/.env created"
elif ! grep -q '^DATABASE_URL=' backend/.env; then
    echo "📝 Adding DATABASE_URL to existing backend/.env..."
    echo "" >> backend/.env
    echo "DATABASE_URL=postgresql+asyncpg://researchpilot:researchpilot@postgres:5432/researchpilot" >> backend/.env
fi

echo "⏳ Waiting for GROBID..."
for i in {1..60}; do
    if curl -sf http://grobid:8070/api/isalive > /dev/null 2>&1; then
        echo "✅ GROBID is ready!"
        break
    fi
    if [ $i -eq 60 ]; then
        echo "⚠️  Warning: GROBID not responding (start the grobid service later)"
    fi
    sleep 2
done

echo "⏳ Waiting for PostgreSQL..."
for i in {1..30}; do
    if python -c "import socket; s=socket.create_connection(('postgres', 5432), 2); s.close()" 2>/dev/null; then
        echo "✅ PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Warning: PostgreSQL not responding (start the postgres service and run alembic later)"
    fi
    sleep 1
done

echo "🐍 Installing Python dependencies with uv..."
cd backend
uv sync --extra dev
cd ..
echo "✅ Python dependencies installed"

echo "🗄️  Running database migrations..."
cd backend
if uv run alembic upgrade head; then
    echo "✅ Database migrations applied"
else
    echo "⚠️  Warning: alembic upgrade failed (is PostgreSQL running?)"
fi
cd ..

echo "📦 Installing frontend dependencies with npm..."
cd frontend
npm install
cd ..
echo "✅ Frontend dependencies installed"

echo "🔧 Installing TypeScript globally..."
sudo npm install -g typescript@5.9.3
echo "✅ TypeScript installed globally"

mkdir -p data/papers

pull_ollama_model() {
    local model="$1"
    echo "🤖 Downloading Ollama model (${model})..."
    if curl -s http://ollama:11434/api/tags | grep -q "${model}"; then
        echo "✅ Model ${model} already exists"
        return
    fi
    curl -X POST http://ollama:11434/api/pull -d "{\"name\":\"${model}\"}" 2>/dev/null &
    PULL_PID=$!
    while kill -0 $PULL_PID 2>/dev/null; do
        echo -n "."
        sleep 2
    done
    echo ""
    echo "✅ Model ${model} downloaded successfully!"
}

if curl -s http://ollama:11434/api/tags > /dev/null 2>&1; then
    pull_ollama_model "gemma3:4b"
    pull_ollama_model "nomic-embed-text"
else
    echo "⚠️  Ollama service not available, skipping model download"
fi

echo ""
echo "📋 Installed versions:"
echo "  Python: $(python --version)"
echo "  Node.js: $(node --version)"
echo "  npm: $(npm --version)"
echo "  TypeScript: $(tsc --version)"
echo "  uv: $(uv --version)"
echo "  Ollama: Running as Docker service at http://ollama:11434"
echo "  GROBID: Running as Docker service at http://grobid:8070"

echo ""
echo "✨ Development environment ready!"
echo ""
echo "📖 To start the app, open TWO terminals:"
echo ""
echo "  Terminal 1:  cd backend && uv run uvicorn app:app --reload --host 0.0.0.0 --port 8000"
echo "  Terminal 2:  cd frontend && npm run dev"
echo "  Browser:     http://localhost:3000"
echo ""
