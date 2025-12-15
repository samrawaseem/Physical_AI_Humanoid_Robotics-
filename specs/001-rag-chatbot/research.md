# Research Summary: RAG Chatbot for Existing Book

## Technology Decisions

### Frontend Framework
- **Decision**: React components for Docusaurus integration
- **Rationale**: Docusaurus is built on React, so using React components ensures seamless integration with MDX pages
- **Alternatives considered**: Vanilla JavaScript, Vue components - rejected because Docusaurus is React-based

### Backend Framework
- **Decision**: FastAPI for backend services
- **Rationale**: FastAPI provides excellent async support, automatic API documentation, and strong typing - ideal for RAG query endpoints
- **Alternatives considered**: Flask, Express.js - FastAPI chosen for better performance and built-in OpenAPI generation

### Vector Database
- **Decision**: Qdrant for vector storage and similarity search
- **Rationale**: Qdrant offers efficient similarity search, good Python integration, and scalable architecture for RAG applications
- **Alternatives considered**: Pinecone, Weaviate, FAISS - Qdrant chosen for open-source nature and performance

### Database for Session Storage
- **Decision**: Neon Postgres for chat session persistence
- **Rationale**: Neon provides serverless Postgres with good performance, scalability, and familiar SQL interface
- **Alternatives considered**: SQLite, MongoDB - Postgres chosen for ACID compliance and relational data handling

### Language Model Integration
- **Decision**: OpenAI API for response generation
- **Rationale**: OpenAI provides reliable, high-quality language models that work well with RAG patterns
- **Alternatives considered**: Local models (Llama), Anthropic API - OpenAI chosen for proven quality and integration simplicity

## Architecture Patterns

### API Design
- **Decision**: REST API with POST endpoints for queries
- **Rationale**: Simple, stateless communication between frontend and backend as specified in requirements
- **Alternatives considered**: WebSocket for real-time communication - REST chosen for simplicity and MVP approach

### Text Selection Integration
- **Decision**: JavaScript event handlers to capture selected text
- **Rationale**: Native browser APIs allow capturing selected text to pass to the backend for context-aware queries
- **Implementation**: Using window.getSelection() API

### Frontend Component Structure
- **Decision**: Modular React components (MessageBubble, MessageInput, LoadingIndicator)
- **Rationale**: Component-based architecture enables reusability and maintainability
- **Implementation**: Functional components with React hooks for state management