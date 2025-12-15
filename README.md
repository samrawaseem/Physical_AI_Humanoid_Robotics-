---
title: My Book Chatbot
emoji: 📚
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
---

# RAG Chatbot for Book Documentation

A Retrieval-Augmented Generation (RAG) chatbot that allows users to ask questions about book content and receive contextually relevant answers using vector search and language models.

## Features

- **RAG-based Question Answering**: Ask questions about book content and get relevant answers
- **Context-Aware Responses**: Provides answers based on selected text or current page context
- **Persistent Sessions**: Chat history is preserved between sessions
- **Responsive Design**: Works on desktop and mobile devices
- **Black-Gold Theme**: Accessible and visually appealing UI
- **Secure Implementation**: Input validation, sanitization, and rate limiting

## Architecture

The application consists of:
- **Frontend**: React-based chat interface embedded in MDX pages
- **Backend**: FastAPI service for processing RAG queries
- **Vector Store**: Qdrant for similarity search on book embeddings
- **Database**: Neon Postgres for session/chat logs
- **LLM**: Cohere for answer generation

## Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- Docker (optional, for containerized deployment)
- Cohere API key
- Qdrant vector database (cloud or local)
- Neon Postgres database

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd frontend
   npm install
   ```

4. **Environment Configuration**
   - Copy `backend/.env.example` to `backend/.env`
   - Copy `frontend/.env.example` to `frontend/.env`
   - Add your API keys and configuration values

5. **Start the Services**
   ```bash
   # Terminal 1: Start Qdrant (if using local instance)
   docker run -p 6333:6333 qdrant/qdrant

   # Terminal 2: Start backend
   cd backend
   uvicorn main:app --reload

   # Terminal 3: Start frontend
   cd frontend
   npm start
   ```

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - Qdrant UI: http://localhost:6333

## Ingestion Pipeline

To add book content to the vector store:

```bash
cd backend
python -m src.scripts.ingest_books <path_to_book_content> --book-id <unique_book_id> --title "Book Title"
```

## API Endpoints

### POST /api/v1/query
Process user queries and return RAG-generated responses

**Request Body**:
```json
{
  "question": "string, required - The user's question about the book content",
  "selected_text": "string, optional - Text selected by the user for context-specific queries",
  "page_content": "string, optional - Current page content for context-aware responses"
}
```

**Response**:
```json
{
  "answer": "string - The generated answer based on book content",
  "sources": [
    {
      "content_snippet": "string - Relevant snippet from book content",
      "page_reference": "string - Reference to the source page",
      "similarity_score": "number - Similarity score of the match"
    }
  ],
  "session_id": "string - Unique identifier for the chat session",
  "message_id": "string - Unique identifier for this message"
}
```

### GET /api/v1/session/{session_id}
Retrieve a chat session with its messages

## Environment Variables

### Backend (.env)
- `QDRANT_URL`: URL for Qdrant vector database
- `QDRANT_API_KEY`: API key for Qdrant (if using cloud)
- `COHERE_API_KEY`: Your Cohere API key
- `DATABASE_URL`: Connection string for Neon Postgres
- `VECTOR_COLLECTION_NAME`: Name of the Qdrant collection (default: book_content)
- `EMBEDDING_DIMENSION`: Dimension of embeddings (default: 1024 for Cohere)

### Frontend (.env)
- `REACT_APP_API_URL`: Backend API URL
- `REACT_APP_API_BASE_URL`: Backend API base URL

## Security Features

- Input validation and sanitization
- Rate limiting to prevent abuse
- Secure API key handling
- XSS protection through content sanitization
- SQL injection prevention through ORM usage

## Error Handling

The system gracefully handles:
- Vector store unavailability
- LLM API errors
- Network timeouts
- Invalid user inputs
- Rate limit exceeded scenarios

## Testing

### Backend Tests
```bash
cd backend
pytest
```

### Frontend Tests
```bash
cd frontend
npm test
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository.
