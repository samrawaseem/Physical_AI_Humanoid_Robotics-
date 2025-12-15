// frontend/src/components/Chatbot/Chatbot.tsx
import React, { useState, useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import MessageInput from './MessageInput';
import LoadingIndicator from './LoadingIndicator';
import useTextSelection from '../../hooks/useTextSelection';
import apiService from '../../services/apiService';
import './Chatbot.css';

interface Message {
    id: string;
    text: string;
    sender: 'user' | 'bot';
    timestamp: Date;
}

const Chatbot: React.FC = () => {
    const [messages, setMessages] = useState<Message[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [sessionId, setSessionId] = useState<string | null>(null);
    const { selectedText } = useTextSelection();
    const messagesEndRef = useRef<HTMLDivElement>(null);

    // Load session from localStorage on component mount
    useEffect(() => {
        const savedSessionId = localStorage.getItem('chatbot_session_id');
        if (savedSessionId) {
            setSessionId(savedSessionId);
            // Optionally load the session history here
            loadSessionHistory(savedSessionId);
        }
    }, []);

    // Auto-scroll to bottom when messages change
    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    const loadSessionHistory = async (sessionId: string) => {
        try {
            // Load the session from the backend
            const sessionData = await apiService.getSession(sessionId);
            // Convert the session data to our local message format
            const loadedMessages: Message[] = sessionData.messages.map((msg: any) => ({
                id: msg.id,
                text: msg.content,
                sender: msg.sender,
                timestamp: new Date(msg.timestamp),
            }));
            setMessages(loadedMessages);
        } catch (error) {
            console.error('Failed to load session history:', error);
            // If loading fails, we'll continue with an empty message list
        }
    };

    const handleSendMessage = async (text: string) => {
        // Add user message to chat
        const userMessage: Message = {
            id: Date.now().toString(),
            text,
            sender: 'user',
            timestamp: new Date(),
        };

        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);

        try {
            // Get current page content for context (this would typically come from props or context)
            const currentPageContent = document.body.innerText.substring(0, 2000); // Simplified approach

            // Call the backend API with the current session ID and page context
            const response = await apiService.query({
                question: text,
                selected_text: selectedText || undefined,
                page_content: currentPageContent || undefined
            }, sessionId || undefined);

            // Update session ID if it changed
            if (response.session_id && response.session_id !== sessionId) {
                setSessionId(response.session_id);
                localStorage.setItem('chatbot_session_id', response.session_id);
            }

            // Add bot response to chat
            const botMessage: Message = {
                id: response.message_id,
                text: response.answer,
                sender: 'bot',
                timestamp: new Date(),
            };

            setMessages(prev => [...prev, botMessage]);
        } catch (error) {
            // Add error message to chat
            const errorMessage: Message = {
                id: Date.now().toString(),
                text: 'Sorry, I encountered an error processing your request. Please try again.',
                sender: 'bot',
                timestamp: new Date(),
            };

            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="chatbot-container">
            <div className="chatbot-header">
                <h3>Book Assistant</h3>
            </div>

            <div className="chatbot-messages">
                {messages.length === 0 ? (
                    <div className="welcome-message">
                        <p>Hello! I'm your book assistant. Ask me anything about the content, and I'll help you find relevant information.</p>
                        {selectedText && (
                            <p className="selected-text-preview">
                                <strong>Selected text:</strong> {selectedText.substring(0, 100)}...
                            </p>
                        )}
                    </div>
                ) : (
                    messages.map((message) => (
                        <MessageBubble
                            key={message.id}
                            text={message.text}
                            sender={message.sender}
                            timestamp={message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        />
                    ))
                )}

                {isLoading && <LoadingIndicator />}
                <div ref={messagesEndRef} />
            </div>

            <div className="chatbot-input">
                <MessageInput onSendMessage={handleSendMessage} disabled={isLoading} />
            </div>
        </div>
    );
};

export default Chatbot;