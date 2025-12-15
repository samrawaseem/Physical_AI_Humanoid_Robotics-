# Quickstart Guide: RAG Chatbot for Existing Book

## Development Setup

### Prerequisites
- Node.js 18+ for Docusaurus frontend
- Python 3.9+ for FastAPI backend
- Docker (for Qdrant vector store)
- Access to Cohere API

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Install backend dependencies**
   ```bash
   cd backend
   pip install fastapi uvicorn python-dotenv cohere qdrant-client
   ```

3. **Install frontend dependencies**
   ```bash
   cd frontend  # or root if using monorepo
   npm install
   ```

4. **Set up environment variables**
   ```bash
   # backend/.env
   COHERE_API_KEY=your_cohere_api_key
   QDRANT_URL=your_qdrant_url
   QDRANT_API_KEY=your_qdrant_api_key
   DATABASE_URL=your_neon_postgres_url
   ```

### Running the Application

1. **Start Qdrant vector store**
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```

2. **Start backend server**
   ```bash
   cd backend
   uvicorn main:app --reload --port 8000
   ```

3. **Start Docusaurus frontend**
   ```bash
   cd frontend
   npm start
   ```

## Key Components

### Frontend Components
- `Chatbot.tsx`: Main chat interface component
- `MessageBubble.tsx`: Individual message display
- `MessageInput.tsx`: Input field with send button
- `LoadingIndicator.tsx`: Visual indicator during API calls

### Backend Endpoints
- `POST /api/query`: Process RAG queries
- Accepts: `{question: string, selected_text?: string}`
- Returns: `{answer: string, sources: array, session_id: string}`

### Database Schema
- `chat_sessions`: Store conversation metadata
- `chat_messages`: Store individual messages
- `book_content`: Store book text segments with embeddings

## Development Workflow

1. **Add new book content**: Run embedding pipeline to update vector store
2. **Modify chat UI**: Update React components in `src/components/chat/`
3. **Change backend logic**: Update FastAPI routes in `backend/api/`
4. **Update tests**: Add/modify tests in respective test directories