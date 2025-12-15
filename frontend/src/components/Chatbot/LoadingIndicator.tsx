// frontend/src/components/Chatbot/LoadingIndicator.tsx
import React from 'react';
import './LoadingIndicator.css';

interface LoadingIndicatorProps {
  message?: string;
}

const LoadingIndicator: React.FC<LoadingIndicatorProps> = ({ message = "Thinking..." }) => {
  return (
    <div className="loading-container">
      <div className="loading-spinner">
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
        <div className="spinner-dot"></div>
      </div>
      <div className="loading-message">{message}</div>
    </div>
  );
};

export default LoadingIndicator;