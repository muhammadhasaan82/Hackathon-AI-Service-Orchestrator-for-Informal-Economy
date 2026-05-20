module.exports = {
  apps: [
    {
      name: "profixer-api",
      cwd: "/home/azureuser/Hackathon-AI-Service-Orchestrator-for-Informal-Economy/backend/generative_ai_project",
      script: "/home/azureuser/Hackathon-AI-Service-Orchestrator-for-Informal-Economy/.venv/bin/uvicorn",
      args: "src.api.app:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      autorestart: true,
      max_memory_restart: "1500M",
      env: {
        PYTHONUNBUFFERED: "1"
      },
      error_file: "./logs/profixer-api-error.log",
      out_file: "./logs/profixer-api-out.log",
      log_file: "./logs/profixer-api-combined.log",
      time: true
    }
  ]
};
