#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# pm2-startup.sh — ProFixer AI Service Orchestrator
# Brings up the full stack:
#   1. Docker Compose (Weaviate, Redis, PostgreSQL)
#   2. Ollama (systemd)
#   3. FastAPI via PM2
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

# ── Variables ─────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
HEALTH_URL="http://127.0.0.1:8000/api/v1/health"
HEALTH_RETRIES=15
HEALTH_DELAY=4

# ── Colored logging helpers ───────────────────────────────────
info()  { printf "\033[0;32m[INFO]\033[0m  %s\n" "$*"; }
warn()  { printf "\033[0;33m[WARN]\033[0m  %s\n" "$*"; }
fail()  { printf "\033[0;31m[FAIL]\033[0m  %s\n" "$*"; }

# ══════════════════════════════════════════════════════════════
# Step 1 — Docker Compose infrastructure
# ══════════════════════════════════════════════════════════════
info "Step 1/5: Starting Docker Compose infrastructure …"
cd "${PROJECT_DIR}"

docker compose up -d

# Print per-service status
for svc in weaviate redis postgres; do
  status=$(docker compose ps --format '{{.State}}' "${svc}" 2>/dev/null || echo "not found")
  if [[ "${status}" == "running" ]]; then
    info "  ✔ ${svc} — ${status}"
  else
    warn "  ✘ ${svc} — ${status}"
  fi
done

# ══════════════════════════════════════════════════════════════
# Step 2 — Verify Ollama
# ══════════════════════════════════════════════════════════════
info "Step 2/5: Checking Ollama service …"

if systemctl is-active --quiet ollama 2>/dev/null; then
  info "  ✔ Ollama systemd service is active"
elif command -v ollama &>/dev/null && ollama list &>/dev/null; then
  info "  ✔ Ollama is reachable (non-systemd)"
else
  fail "  ✘ Ollama is NOT running"
  warn "  → Try: sudo systemctl start ollama"
fi

# Ensure the required model is available
if ollama list 2>/dev/null | grep -q "gemma3:1b"; then
  info "  ✔ Model gemma3:1b is available"
else
  warn "  ✘ Model gemma3:1b not found — pulling now …"
  ollama pull gemma3:1b
  info "  ✔ Model gemma3:1b pulled successfully"
fi

# ══════════════════════════════════════════════════════════════
# Step 3 — Ensure logs directory
# ══════════════════════════════════════════════════════════════
info "Step 3/5: Ensuring logs directory …"
mkdir -p "${PROJECT_DIR}/logs"
info "  ✔ ${PROJECT_DIR}/logs"

# ══════════════════════════════════════════════════════════════
# Step 4 — Start FastAPI via PM2
# ══════════════════════════════════════════════════════════════
info "Step 4/5: Starting FastAPI via PM2 …"
cd "${PROJECT_DIR}"

# Delete old instance if present (ignore errors)
pm2 delete profixer-api 2>/dev/null || true

pm2 start ecosystem.config.js
pm2 save --force

info "  ✔ PM2 process started"

# ══════════════════════════════════════════════════════════════
# Step 5 — Health check
# ══════════════════════════════════════════════════════════════
info "Step 5/5: Waiting for FastAPI health endpoint …"
info "  URL: ${HEALTH_URL}"

healthy=false
for i in $(seq 1 "${HEALTH_RETRIES}"); do
  if response=$(curl -sf --max-time 5 "${HEALTH_URL}" 2>/dev/null); then
    info "  ✔ Health check passed (attempt ${i}/${HEALTH_RETRIES})"
    info "  Response: ${response}"
    healthy=true
    break
  fi
  warn "  … attempt ${i}/${HEALTH_RETRIES} failed — retrying in ${HEALTH_DELAY}s"
  sleep "${HEALTH_DELAY}"
done

if [[ "${healthy}" != "true" ]]; then
  fail "  ✘ Health check failed after ${HEALTH_RETRIES} attempts"
  fail "  Recent PM2 logs:"
  pm2 logs profixer-api --lines 30 --nostream || true
  echo ""
  warn "Troubleshooting tips:"
  warn "  1. pm2 logs profixer-api --lines 50"
  warn "  2. docker compose ps"
  warn "  3. ollama list"
  warn "  4. ss -tlnp | grep -E '8000|8080|6379|5433|11434'"
  exit 1
fi

# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
echo ""
info "════════════════════════════════════════════════════════"
info " ProFixer API — All systems operational"
info "════════════════════════════════════════════════════════"
pm2 status
echo ""
info "Useful commands:"
info "  pm2 logs profixer-api    — live logs"
info "  pm2 monit                — terminal dashboard"
info "  pm2 restart profixer-api — restart API"
