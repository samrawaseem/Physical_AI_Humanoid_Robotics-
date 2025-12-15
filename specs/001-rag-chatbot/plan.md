# Implementation Plan: RAG Chatbot for Existing Book

**Branch**: `001-rag-chatbot` | **Date**: 2025-12-10 | **Spec**: [specs/001-rag-chatbot/spec.md](specs/001-rag-chatbot/spec.md)
**Input**: Feature specification from `/specs/001-rag-chatbot/spec.md`

**Note**: This template is filled in by the `/sp.plan` command. See `.specify/templates/commands/plan.md` for the execution workflow.

## Summary

Implementation of a RAG (Retrieval Augmented Generation) chatbot for Docusaurus-based book documentation. The solution includes a React-based chat interface embedded in MDX pages, a FastAPI backend for processing queries against a Qdrant vector store, and Neon Postgres for session management. The system enables users to query book content via natural language and supports both full-book and selected-text queries with a responsive, accessible black-gold themed UI.

## Technical Context

**Language/Version**: Python 3.9+ (backend), JavaScript/TypeScript (frontend), Node.js 18+
**Primary Dependencies**: FastAPI, React, Qdrant-client, OpenAI, Neon Postgres
**Storage**: Neon Postgres for session/chat logs, Qdrant vector store for book embeddings
**Testing**: pytest (backend), Jest/React Testing Library (frontend)
**Target Platform**: Web browser (Docusaurus integration)
**Project Type**: Web application (frontend + backend)
**Performance Goals**: <5s response time for queries, 90%+ accuracy on relevant queries
**Constraints**: REST POST requests, responsive and accessible UI, black-gold theme
**Scale/Scope**: Single book with multiple chapters, multiple concurrent users

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **AI-Native Workflow**: ✅ Using Spec-Kit Plus and Claude Code for all development tasks
- **Specification-Driven Development**: ✅ Following specification from spec.md
- **Technical Accuracy**: ✅ Using appropriate technologies for RAG implementation
- **Clarity and Accessibility**: ✅ Designing accessible chat UI as required
- **Consistency and Standards**: ✅ Following Docusaurus conventions for integration
- **Transparent Version Control**: ✅ All changes tracked in Git

## Project Structure

### Documentation (this feature)

```text
specs/001-rag-chatbot/
├── plan.md              # This file (/sp.plan command output)
├── research.md          # Phase 0 output (/sp.plan command)
├── data-model.md        # Phase 1 output (/sp.plan command)
├── quickstart.md        # Phase 1 output (/sp.plan command)
├── contracts/           # Phase 1 output (/sp.plan command)
│   └── query-api.yaml
└── tasks.md             # Phase 2 output (/sp.tasks command - NOT created by /sp.plan)
```

### Source Code (repository root)

```text
backend/
├── src/
│   ├── models/
│   │   ├── chat_session.py
│   │   └── chat_message.py
│   ├── services/
│   │   ├── rag_service.py
│   │   ├── qdrant_service.py
│   │   └── openai_service.py
│   └── api/
│       └── v1/
│           └── query.py
├── main.py
└── requirements.txt

frontend/
├── src/
│   ├── components/
│   │   ├── Chatbot/
│   │   │   ├── Chatbot.tsx
│   │   │   ├── MessageBubble.tsx
│   │   │   ├── MessageInput.tsx
│   │   │   └── LoadingIndicator.tsx
│   │   └── hooks/
│   │       └── useTextSelection.ts
│   ├── services/
│   │   └── apiService.ts
│   └── styles/
│       └── chatbot.css
├── static/
│   └── img/
└── package.json

contracts/
└── query-api.yaml

tests/
├── backend/
│   ├── unit/
│   └── integration/
└── frontend/
    ├── unit/
    └── integration/
```

**Structure Decision**: Web application structure with separate backend (FastAPI) and frontend (React/Docusaurus) components to maintain clear separation of concerns while enabling seamless integration with Docusaurus via React components embedded in MDX pages.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple services | RAG requires vector store, LLM, and session management | Single service would create tight coupling and scalability issues |
| Separate backend/frontend | Docusaurus requires React components but needs robust backend for RAG | Monolithic approach would limit scalability and maintainability |
