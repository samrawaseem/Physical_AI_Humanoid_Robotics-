// frontend/src/services/apiService.ts
import axios, { AxiosResponse } from 'axios';
import API_CONFIG from '../config/apiConfig';

// Define types for our API requests and responses
interface QueryRequest {
  question: string;
  selected_text?: string;
  page_content?: string;
}

interface Source {
  content_snippet: string;
  page_reference: string;
  similarity_score: number;
}

interface QueryResponse {
  answer: string;
  sources: Source[];
  session_id: string;
  message_id: string;
}

interface SessionResponse {
  id: string;
  created_at: string;
  updated_at: string;
  user_id: string;
  messages: Array<{
    id: string;
    session_id: string;
    sender: string;
    content: string;
    timestamp: string;
    context: any;
  }>;
}

interface ErrorResponse {
  error: string;
  code: string;
}

class ApiService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_CONFIG.BASE_URL;
  }

  async query(request: QueryRequest, sessionId?: string): Promise<QueryResponse> {
    try {
      const headers: any = {
        'Content-Type': 'application/json',
      };

      // Add session ID to the URL if provided
      let url = `${this.baseUrl}/query`;
      if (sessionId) {
        url += `?session_id=${sessionId}`;
      }

      const response: AxiosResponse<QueryResponse> = await axios.post(
        url,
        request,
        {
          headers,
        }
      );
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const errorResponse: ErrorResponse = {
          error: error.response?.data?.error || 'Unknown error occurred',
          code: error.response?.data?.code || 'UNKNOWN_ERROR',
        };
        throw new Error(errorResponse.error);
      }
      throw new Error('Network error occurred');
    }
  }

  async getSession(sessionId: string): Promise<SessionResponse> {
    try {
      const response = await axios.get(`${this.baseUrl}/session/${sessionId}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || 'Failed to fetch session');
      }
      throw new Error('Network error occurred');
    }
  }
}

export default new ApiService();