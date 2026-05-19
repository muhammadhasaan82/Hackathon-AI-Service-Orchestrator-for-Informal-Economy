# Deployment Guide — ProFixer AI Service Orchestrator

> Production runtime on Azure VM using **PM2** (FastAPI), **Docker Compose** (infrastructure), and **systemd** (Ollama).

---

## Architecture

```
PM2 ─────── FastAPI / Uvicorn  (:8000)
Docker ──── Weaviate           (:8080, gRPC :50051)
         ── Redis              (:6379)
         ── PostgreSQL         (:5433 → 5432)
systemd ─── Ollama             (:11434)
```

---

## Prerequisites

| Component            | Version / Notes                            |
| -------------------- | ------------------------------------------ |
| **Node.js**          | 18+ (needed for PM2)                       |
| **PM2**              | `npm install -g pm2`                       |
| **Docker + Compose** | Docker Engine 24+, Compose v2              |
| **Python**           | 3.12 with virtual env at `.venv/`          |
| **Ollama**           | Installed as systemd service               |
| **LLM Model**        | `gemma3:1b` pulled (`ollama pull gemma3:1b`) |

Ensure the `.env` file exists at `backend/generative_ai_project/.env` with the correct connection strings (see `.env.example`).

---

## Quick Start

```bash
cd backend/generative_ai_project

# Make the startup script executable
chmod +x pm2-startup.sh

# Run the full startup sequence
./pm2-startup.sh
```

The script will:

1. Start Docker Compose services (Weaviate, Redis, PostgreSQL)
2. Verify Ollama is running and the `gemma3:1b` model is available
3. Create the `logs/` directory
4. Start FastAPI via PM2
5. Run a health-check loop against `http://127.0.0.1:8000/api/v1/health`

---

## PM2 Auto-Start on Boot

```bash
# Generate the systemd startup script (run the printed sudo command)
pm2 startup

# Save the current PM2 process list so it restores on reboot
pm2 save
```

---

## Monitoring

```bash
# Process table
pm2 status

# Live log stream
pm2 logs profixer-api

# Terminal dashboard (CPU, memory, logs)
pm2 monit
```

---

## Restart / Stop

```bash
# Graceful restart
pm2 restart profixer-api

# Stop without removing from PM2
pm2 stop profixer-api

# Remove from PM2 entirely
pm2 delete profixer-api
```

---

## Infrastructure Services

### Docker (Weaviate, Redis, PostgreSQL)

```bash
# Start / ensure running
docker compose up -d

# Status
docker compose ps

# Follow logs
docker compose logs -f

# Restart a single service
docker compose restart redis
```

### Ollama

```bash
# Status
sudo systemctl status ollama

# Restart
sudo systemctl restart ollama

# List available models
ollama list

# Pull / update a model
ollama pull gemma3:1b
```

---

## Troubleshooting

| Symptom                    | Diagnostic Command                                            |
| -------------------------- | ------------------------------------------------------------- |
| API not responding         | `pm2 logs profixer-api --lines 50`                            |
| Health check fails         | `curl -v http://127.0.0.1:8000/api/v1/health`                |
| Docker service down        | `docker compose ps`                                           |
| Ollama model missing       | `ollama list`                                                 |
| Port conflict              | `ss -tlnp \| grep -E '8000\|8080\|6379\|5433\|11434'`        |
| PM2 process crash-looping  | `pm2 describe profixer-api` — check restart count & status    |
| Memory issues              | `pm2 monit` — watch RSS column (limit: 1500 MB)              |

### Log Locations

| Log                     | Path                                |
| ----------------------- | ----------------------------------- |
| PM2 stdout              | `./logs/pm2-out.log`                |
| PM2 stderr              | `./logs/pm2-error.log`              |
| PM2 daemon              | `~/.pm2/pm2.log`                    |
| Docker service logs     | `docker compose logs <service>`     |
| Ollama                  | `journalctl -u ollama`              |

---

## Smoke Test

```bash
python scripts/smoke_api.py --base-url http://127.0.0.1:8000
```

---

## Environment Variables

The FastAPI app loads its `.env` file via `python-dotenv` / `pydantic-settings`. Key variables:

| Variable           | Default                        | Description                 |
| ------------------ | ------------------------------ | --------------------------- |
| `APP_HOST`         | `0.0.0.0`                     | Bind address                |
| `APP_PORT`         | `8000`                         | Bind port                   |
| `LOG_LEVEL`        | `INFO`                         | Python logging level        |
| `OLLAMA_BASE_URL`  | `http://localhost:11434`       | Ollama API endpoint         |
| `OLLAMA_MODEL`     | `gemma3:1b`                    | Default LLM model           |
| `WEAVIATE_URL`     | `http://localhost:8080`        | Weaviate REST endpoint      |
| `REDIS_URL`        | `redis://localhost:6379/0`     | Redis connection string     |
| `DATABASE_URL`     | `postgresql://…@localhost:5432`| PostgreSQL connection       |

See `.env.example` for the full list.
