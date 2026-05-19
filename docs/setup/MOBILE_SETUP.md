# Mobile Setup

Mobile source lives in:

```text
mobile/ProFixer
```

This is the Flutter client for the AI Service Orchestrator backend.

## Placeholder Local Commands

Run these later in a machine with Flutter installed:

```bash
cd mobile/ProFixer
flutter pub get
flutter run
```

## Backend API URL Note

The app should point to the deployed backend API and WebSocket URLs when running against GCP. Check `mobile/ProFixer/lib/main.dart` for the backend client configuration.

For a future run, configure the API URL with Dart defines such as:

```bash
flutter run --dart-define=PROFIXER_API_BASE_URL=https://YOUR_GCP_BACKEND_URL
```

If WebSocket is hosted separately or needs an explicit scheme, also configure `PROFIXER_WS_BASE_URL`.
