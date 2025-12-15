// frontend/tests/unit/Chatbot.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import Chatbot from '../../src/components/Chatbot/Chatbot';

// Mock the API service
jest.mock('../../src/services/apiService', () => ({
  query: jest.fn().mockResolvedValue({
    answer: 'Test answer from API',
    sources: [{ content_snippet: 'Test snippet', page_reference: 'page_1', similarity_score: 0.9 }],
    session_id: 'test-session-id',
    message_id: 'test-message-id'
  })
}));

// Mock the useTextSelection hook
jest.mock('../../src/hooks/useTextSelection', () => ({
  default: () => ({
    selectedText: '',
    selectionInfo: null,
    getSelectedText: jest.fn(),
    getSelectionInfo: jest.fn(),
    clearSelection: jest.fn()
  })
}));

describe('Chatbot Component', () => {
  test('renders without crashing', () => {
    render(<Chatbot />);
    expect(screen.getByText('Book Assistant')).toBeInTheDocument();
  });

  test('displays welcome message when no messages exist', () => {
    render(<Chatbot />);
    expect(screen.getByText(/Hello! I'm your book assistant/i)).toBeInTheDocument();
  });

  test('allows user to send a message', async () => {
    render(<Chatbot />);

    const input = screen.getByPlaceholderText('Type your question here...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.click(sendButton);

    // Wait for the message to be processed
    await waitFor(() => {
      expect(screen.getByText('Test question')).toBeInTheDocument();
    });
  });

  test('displays loading indicator when waiting for response', async () => {
    render(<Chatbot />);

    const input = screen.getByPlaceholderText('Type your question here...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.click(sendButton);

    // Check that loading indicator appears
    expect(screen.getByText('Thinking...')).toBeInTheDocument();
  });

  test('displays bot response after API call', async () => {
    render(<Chatbot />);

    const input = screen.getByPlaceholderText('Type your question here...');
    const sendButton = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Test question' } });
    fireEvent.click(sendButton);

    // Wait for the bot response to appear
    await waitFor(() => {
      expect(screen.getByText('Test answer from API')).toBeInTheDocument();
    });
  });
});