# Project Structure

This repository is organized as a small monorepo so hackathon reviewers and future contributors can quickly understand each application boundary.

## Backend

```text
backend/generative_ai_project
```

Contains the Python/FastAPI agentic AI service orchestrator, including agents, API routes, configuration, RAG components, evaluation files, tests, Dockerfile, docker-compose file, and environment examples.

## Mobile

```text
mobile/ProFixer
```

Contains the Flutter application used as the client experience for service discovery, AI chat, and provider browsing.

## Frontend

```text
frontend
```

Reserved for a future dedicated web frontend. It is intentionally separate from Flutter web support, which remains inside the mobile project.

## Docs

```text
docs
```

Contains hackathon challenge files, architecture references, and setup guides.

## Scripts

```text
scripts
```

Reserved for repo-level helper scripts that are shared across backend, mobile, frontend, or deployment workflows.
