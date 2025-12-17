// Configuration for API endpoints
// Automatically detects environment:
// - Local development (localhost): uses local backend at localhost:8000
// - Production (Vercel): uses Hugging Face backend
// - Can be overridden with REACT_APP_API_BASE_URL environment variable

const isLocalhost = typeof window !== 'undefined' &&
  (window.location.hostname === 'localhost' ||
    window.location.hostname === '127.0.0.1' ||
    window.location.hostname === '');

const getApiBaseUrl = (): string => {
  // First priority: Environment variable override
  if (typeof process !== 'undefined' && process.env) {
    const envUrl = process.env.REACT_APP_API_BASE_URL || process.env.API_BASE_URL;
    if (envUrl) {
      return envUrl;
    }
  }

  // Second priority: Auto-detect based on hostname
  if (isLocalhost) {
    return 'http://localhost:8000/api/v1';
  } else {
    return 'https://samra09-swbook.hf.space/api/v1';
  }
};

const API_CONFIG = {
  BASE_URL: getApiBaseUrl(),
};

export default API_CONFIG;