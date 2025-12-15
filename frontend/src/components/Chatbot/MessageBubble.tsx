// frontend/src/components/Chatbot/MessageBubble.tsx
import React from 'react';
import './MessageBubble.css';

interface MessageBubbleProps {
  text: string;
  sender: 'user' | 'bot';
  timestamp?: string;
}

const MessageBubble: React.FC<MessageBubbleProps> = ({ text, sender, timestamp }) => {
  return (
    <div className={`message-bubble ${sender}`}>
      <div className="message-content">{text}</div>
      {timestamp && <div className="message-timestamp">{timestamp}</div>}
    </div>
  );
};

export default MessageBubble;