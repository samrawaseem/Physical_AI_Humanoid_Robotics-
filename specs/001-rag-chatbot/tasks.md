# Tasks: RAG Chatbot for Existing Book

**Feature**: RAG Chatbot for Existing Book | **Branch**: 001-rag-chatbot
**Input**: Feature specification, implementation plan, and contracts from `/specs/001-rag-chatbot/`

## Implementation Strategy

MVP scope includes User Story 1 (P1) with basic chat functionality, backend API, and Qdrant integration. Subsequent user stories (P2, P3) will be implemented incrementally.

## Dependencies

- User Story 1 (P1) must be completed before User Story 2 (P2) and User Story 3 (P3)
- Backend foundational tasks must complete before frontend implementation
- Database and vector store setup required before API implementation

## Parallel Execution Examples

- Frontend component development can run parallel to backend service development
- Database models can be developed in parallel with API endpoint implementation
- Styling and accessibility features can be added in parallel to core functionality

---

## Phase 1: Setup

**Goal**: Initialize project structure and development environment

- [x] T001 Create project structure per implementation plan: backend/, frontend/, contracts/, tests/
- [x] T002 Initialize backend requirements.txt with FastAPI, Qdrant-client, OpenAI, SQLAlchemy, asyncpg dependencies
- [x] T003 Initialize frontend package.json with React, Docusaurus dependencies
- [x] T004 Create backend configuration module in backend/src/config.py
- [x] T005 [P] Create backend main application file in backend/main.py
- [x] T006 [P] Create frontend basic structure in frontend/src/components/
- [x] T007 Create backend .env file with environment variable placeholders
- [x] T008 Create frontend .env file with API endpoint configuration
- [x] T009 Setup gitignore for sensitive files (.env, __pycache__, node_modules)

## Phase 2: Foundational

**Goal**: Establish core infrastructure and services needed by all user stories

- [x] T010 Setup database models for ChatSession in backend/src/models/chat_session.py
- [x] T011 Setup database models for ChatMessage in backend/src/models/chat_message.py
- [x] T012 Create database connection utilities in backend/src/database.py
- [x] T013 [P] Setup Qdrant client service in backend/src/services/qdrant_service.py
- [x] T014 [P] Setup OpenAI service in backend/src/services/openai_service.py
- [x] T015 [P] Setup RAG service in backend/src/services/rag_service.py
- [x] T016 Create API error handlers in backend/src/api/errors.py
- [x] T017 Setup database migrations with Alembic in backend/alembic/
- [x] T018 Create frontend API service in frontend/src/services/apiService.ts
- [x] T019 [P] Create frontend hooks for text selection in frontend/src/hooks/useTextSelection.ts

## Phase 3: User Story 1 - Query Book Content via Chat Interface (Priority: P1)

**Goal**: Enable users to ask questions about book content through a chat interface

**Independent Test**: Can be fully tested by entering questions about book content and verifying that the chatbot provides relevant, accurate responses based on the book's text.

**Acceptance Scenarios**:
1. **Given** I am viewing a book page with an embedded chatbot, **When** I type a question about the book content, **Then** the chatbot responds with relevant information from the book.
2. **Given** I have selected specific text in the book, **When** I ask a question about that text, **Then** the chatbot provides contextually relevant answers focused on the selected content.

- [x] T020 [US1] Create POST /api/query endpoint in backend/src/api/v1/query.py
- [x] T021 [US1] Implement request validation for QueryRequest in backend/src/api/v1/query.py
- [x] T022 [US1] Implement response formatting for QueryResponse in backend/src/api/v1/query.py
- [x] T023 [US1] [P] Create MessageBubble component in frontend/src/components/Chatbot/MessageBubble.tsx
- [x] T024 [US1] [P] Create MessageInput component in frontend/src/components/Chatbot/MessageInput.tsx
- [x] T025 [US1] [P] Create LoadingIndicator component in frontend/src/components/Chatbot/LoadingIndicator.tsx
- [x] T026 [US1] Create Chatbot main component in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T027 [US1] Implement frontend API call to POST /api/query in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T028 [US1] Implement text selection capture functionality in frontend/src/hooks/useTextSelection.ts
- [x] T029 [US1] [P] Create black-gold theme CSS in frontend/src/styles/chatbot.css
- [x] T030 [US1] Connect frontend to backend API in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T031 [US1] Implement auto-scroll functionality in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T032 [US1] [P] Create endpoint tests for /api/query in backend/tests/integration/test_query.py
- [x] T033 [US1] [P] Create component tests for Chatbot in frontend/tests/unit/Chatbot.test.tsx
- [x] T034 [US1] Test basic question answering functionality

## Phase 4: User Story 2 - Persistent Chat Sessions (Priority: P2)

**Goal**: Preserve chat history between sessions to enable continuity of conversations

**Independent Test**: Can be tested by starting a conversation, closing the browser, returning to the book, and verifying that previous chat history is available.

**Acceptance Scenarios**:
1. **Given** I have participated in a chat session, **When** I return to the book later, **Then** my previous chat history is accessible and preserved.

- [x] T035 [US2] Implement ChatSession CRUD operations in backend/src/services/session_service.py
- [x] T036 [US2] Update POST /api/query to create/update sessions in backend/src/api/v1/query.py
- [x] T037 [US2] Create GET /api/session/{session_id} endpoint in backend/src/api/v1/session.py
- [x] T038 [US2] Implement session retrieval in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T039 [US2] Add session persistence to local storage in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T040 [US2] [P] Create session tests in backend/tests/integration/test_session.py
- [x] T041 [US2] [P] Update frontend to handle session IDs in API calls
- [x] T042 [US2] Test session persistence across browser sessions

## Phase 5: User Story 3 - Context-Aware Responses (Priority: P3)

**Goal**: Enable the chatbot to understand the context of the current page for more relevant responses

**Independent Test**: Can be tested by asking questions on different pages and verifying that the chatbot appropriately considers or ignores the current page context based on the query.

**Acceptance Scenarios**:
1. **Given** I am viewing a specific page in the book, **When** I ask a contextual question, **Then** the chatbot uses the current page content as additional context for the response.

- [x] T043 [US3] Update POST /api/query to accept page context in backend/src/api/v1/query.py
- [x] T044 [US3] Modify RAG service to consider page context in backend/src/services/rag_service.py
- [x] T045 [US3] Update frontend to pass current page context in API calls
- [x] T046 [US3] [P] Create context-aware tests in backend/tests/integration/test_context.py
- [x] T047 [US3] Update Chatbot component to handle page context in frontend/src/components/Chatbot/Chatbot.tsx
- [x] T048 [US3] Test context-aware response functionality

## Phase 6: Ingestion Pipeline

**Goal**: Implement pipeline to embed book content into Qdrant vector store

- [x] T049 Create book content ingestion script in backend/src/scripts/ingest_books.py
- [x] T050 Implement text segmentation logic in backend/src/services/text_segmentation.py
- [x] T051 Create embedding generation service in backend/src/services/embedding_service.py
- [x] T052 [P] Add BookContent model in backend/src/models/book_content.py
- [x] T053 Implement Qdrant collection setup in backend/src/services/qdrant_service.py
- [x] T054 [P] Create ingestion tests in backend/tests/unit/test_ingestion.py
- [x] T055 Test book content ingestion pipeline end-to-end

## Phase 7: Polish & Cross-Cutting Concerns

**Goal**: Add finishing touches, error handling, and optimize for production

- [x] T056 Implement error handling for vector store unavailability
- [x] T057 Add rate limiting to API endpoints
- [x] T058 Implement proper logging throughout the application
- [x] T059 Add comprehensive input validation and sanitization
- [x] T060 [P] Create Docker configuration for backend in backend/Dockerfile
- [x] T061 [P] Create Docker configuration for frontend
- [ ] T062 [P] Add comprehensive frontend tests in frontend/tests/
- [ ] T063 [P] Add comprehensive backend tests in backend/tests/
- [ ] T064 [P] Add accessibility features to frontend components
- [ ] T065 [P] Optimize frontend bundle size and performance
- [ ] T066 [P] Add documentation for API endpoints
- [ ] T067 [P] Add documentation for frontend components
- [x] T068 [P] Create deployment scripts for backend and frontend
- [ ] T069 [P] Add monitoring and metrics collection
- [ ] T070 [P] Complete responsive design for all screen sizes
- [ ] T071 Final integration testing and bug fixes