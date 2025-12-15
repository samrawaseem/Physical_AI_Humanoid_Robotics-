// src/components/Chatbot/FloatingChatbotButton.tsx
import React, { useState } from 'react';
import Chatbot from './Chatbot';
import './FloatingChatbotButton.css';

const FloatingChatbotButton: React.FC = () => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [isHovered, setIsHovered] = useState<boolean>(false);

  const toggleChatbot = () => {
    setIsOpen(!isOpen);
  };

  return (
    <div className="floating-chatbot">
      {isOpen ? (
        <div className="chatbot-window">
          <Chatbot />
          <button
            className="close-chatbot-button"
            onClick={toggleChatbot}
            aria-label="Close chatbot"
          >
            ×
          </button>
        </div>
      ) : (
        <button
          className={`floating-chatbot-button ${isHovered ? 'hovered' : ''}`}
          onClick={toggleChatbot}
          onMouseEnter={() => setIsHovered(true)}
          onMouseLeave={() => setIsHovered(false)}
          aria-label="Open chatbot"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="chatbot-icon"
          >
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
        </button>
      )}
    </div>
  );
};

export default FloatingChatbotButton;