---
title: RAG Chatbot API
emoji: 🤖
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
---

# RAG Chatbot API

This is the backend API for the Physical AI Humanoid Robotics book RAG chatbot.

## Local Deployment

To build and run this container locally:

```bash
docker build -t rag-backend .
docker run -p 7860:7860 -e PORT=7860 rag-backend
```

## API Documentation

Once running, the API is available at the root URL. 
- Health check: `/health`
- API Docs: `/docs` (if enabled in FastAPI)
