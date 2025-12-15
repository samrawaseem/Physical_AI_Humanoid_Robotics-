// frontend/src/components/Chatbot/MessageInput.tsx
import React, { useState, KeyboardEvent } from 'react';
import './MessageInput.css';

interface MessageInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

const MessageInput: React.FC<MessageInputProps> = ({ onSendMessage, disabled = false }) => {
  const [inputValue, setInputValue] = useState<string>('');

  const handleSubmit = () => {
    if (inputValue.trim() && !disabled) {
      onSendMessage(inputValue.trim());
      setInputValue('');
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="message-input-container">
      <input
        type="text"
        className="message-input"
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Type your question here..."
        disabled={disabled}
      />
      <button
        className="send-button"
        onClick={handleSubmit}
        disabled={disabled || !inputValue.trim()}
      >
        Send
      </button>
    </div>
  );
};

export default MessageInput;