// Configuration for API endpoints
const API_CONFIG = {
  BASE_URL: (typeof process !== 'undefined' && process.env)
    ? (process.env.REACT_APP_API_BASE_URL || process.env.API_BASE_URL || 'http://localhost:8000/api/v1')
    : 'http://localhost:8000/api/v1',
};

export default API_CONFIG;