# Data Model: RAG Chatbot for Existing Book

## Entities

### ChatSession
- **Description**: Represents a user's conversation with the chatbot
- **Fields**:
  - `id` (string/UUID): Unique identifier for the session
  - `created_at` (timestamp): When the session was created
  - `updated_at` (timestamp): When the session was last updated
  - `user_id` (string/UUID, optional): Identifier for authenticated users
  - `metadata` (JSON): Additional session-specific data
- **Relationships**: Contains multiple ChatMessage entities
- **Validation**: Must have a creation timestamp

### ChatMessage
- **Description**: Individual message within a session
- **Fields**:
  - `id` (string/UUID): Unique identifier for the message
  - `session_id` (string/UUID): Reference to parent ChatSession
  - `sender` (enum: "user" | "bot"): Indicates message origin
  - `content` (string): The actual message content
  - `timestamp` (timestamp): When the message was sent
  - `context` (JSON, optional): Additional context (e.g., selected text)
- **Relationships**: Belongs to one ChatSession
- **Validation**: Must have a sender type and content

### BookContent
- **Description**: Segments of book text processed and stored as embeddings
- **Fields**:
  - `id` (string/UUID): Unique identifier for the content segment
  - `content` (string): The text content of the segment
  - `embedding` (vector): Vector representation for similarity search
  - `page_reference` (string, optional): Reference to the source page
  - `section_title` (string, optional): Title of the section
  - `metadata` (JSON): Additional information about the content
- **Relationships**: Used for similarity search during queries
- **Validation**: Must have content and embedding

### QueryResult
- **Description**: Search results from the vector store that inform the chatbot's responses
- **Fields**:
  - `id` (string/UUID): Unique identifier for the result
  - `query` (string): The original query that generated this result
  - `retrieved_content` (array of BookContent): Content retrieved from vector store
  - `similarity_scores` (array of numbers): Similarity scores for each retrieved content
  - `timestamp` (timestamp): When the query was processed
- **Relationships**: Links to BookContent entities that were retrieved
- **Validation**: Must have query and retrieved content

## State Transitions

### ChatSession
- Active (new session created) → Inactive (session not accessed for extended period)
- Can be extended when new messages are added

### ChatMessage
- Pending (message sent, waiting for response) → Processed (response received)
- Context can be updated if additional information is provided