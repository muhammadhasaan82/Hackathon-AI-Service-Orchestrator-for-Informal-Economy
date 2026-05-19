// ═══════════════════════════════════════════════════════════════
// PM2 Ecosystem Config — ProFixer AI Service Orchestrator
// Manages: FastAPI / Uvicorn only
// Docker Compose manages: Weaviate, Redis, PostgreSQL
// systemd manages: Ollama
// ═══════════════════════════════════════════════════════════════

module.exports = {
  apps: [
    {
      name: 'profixer-api',

      // ── Interpreter ────────────────────────────────────────
      // Using the virtual-env Python directly.
      // If you are NOT using a venv, change this to 'python3'
      // and ensure uvicorn is installed globally.
      script: '.venv/bin/python',
      args: '-m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --workers 1',

      // ── Working directory ──────────────────────────────────
      cwd: __dirname,

      // ── Process management ─────────────────────────────────
      autorestart: true,
      watch: false,
      max_memory_restart: '1500M',
      max_restarts: 10,
      restart_delay: 5000,
      kill_timeout: 8000,

      // ── Logging ────────────────────────────────────────────
      out_file: './logs/pm2-out.log',
      error_file: './logs/pm2-error.log',
      merge_logs: true,
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',

      // ── Environment variables ──────────────────────────────
      // These supplement (not replace) the .env file loaded by
      // the FastAPI app via python-dotenv / pydantic-settings.
      env: {
        APP_HOST: '0.0.0.0',
        APP_PORT: '8000',
        LOG_LEVEL: 'INFO',
        ENVIRONMENT: 'production',
      },
    },
  ],
};
