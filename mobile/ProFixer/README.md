# ProFixer Mobile

Flutter mobile client for the AI Service Orchestrator backend.

## Backend Configuration

The app reads backend URLs from Dart defines so the same build can point at
local development or the GCP deployment:

```bash
flutter run \
  --dart-define=PROFIXER_API_BASE_URL=https://YOUR_GCP_BACKEND_URL \
  --dart-define=PROFIXER_WS_BASE_URL=wss://YOUR_GCP_BACKEND_URL
```

For Android emulator local testing, the default API URL is
`http://10.0.2.2:8000` and the WebSocket URL is derived as
`ws://10.0.2.2:8000`.

The chat widget tries WebSocket first at `/api/v1/ws/chat`, then falls back to
REST `/api/v1/chat`. Provider cards use `/api/v1/providers`.
