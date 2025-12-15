import React from 'react';
import FloatingChatbotButton from '../components/Chatbot/FloatingChatbotButton';

// Root component for Docusaurus that wraps the entire application
const Root = ({ children }: { children: React.ReactNode }) => {
    return (
        <>
            {children}
            <FloatingChatbotButton />
        </>
    );
};

export default Root;
