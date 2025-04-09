// API Configuration
const API_CONFIG = {
  // Base URL for the backend API
//   BASE_URL: 'http://localhost:8000',
  BASE_URL: 'http://REDACTED_PUBLIC_IP_2:80',

  // API Endpoints
  ENDPOINTS: {
    MOVIES: '/api/movies',
    PROCESS: '/api/process',
    MOVIE_CLICKED: '/api/movie-clicked',
  },
  
  // TMDB API Configuration
  TMDB: {
    API_KEY: 'REDACTED_TMDB_API_KEY',
    API_URL: 'https://api.themoviedb.org/3/search/movie',
    IMAGE_BASE_URL: 'https://image.tmdb.org/t/p/w500',
  },
};

export default API_CONFIG; 