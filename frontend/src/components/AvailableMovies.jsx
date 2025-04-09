import React, { useState } from 'react';
import { toast } from 'react-toastify';
import './MovieCard.css';
import './AvailableMovies.css';
import API_CONFIG from '../config';

const AvailableMovies = () => {
  const [availableMovies, setAvailableMovies] = useState([]);
  const [movieDetails, setMovieDetails] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedMovie, setSelectedMovie] = useState(null);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [email, setEmail] = useState('');
  const [editLoading, setEditLoading] = useState(false);

  const fetchAvailableMovies = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.MOVIES}`);
      if (!response.ok) {
        throw new Error('Failed to fetch movies');
      }
      const data = await response.json();
      setAvailableMovies(data);
      toast.success(`Found ${data.length} available movies`);
      
      // Search for each movie in TMDB
      const moviePromises = data.map(movieFile => searchMovieInTMDB(movieFile.name));
      const results = await Promise.all(moviePromises);
      console.log(results);
      // Filter out movies that weren't found in TMDB
      const validResults = results.filter(result => result !== null);
      setMovieDetails(validResults);
      
      if (validResults.length < data.length) {
        toast.info(`Found details for ${validResults.length} out of ${data.length} movies`);
      }
    } catch (error) {
      console.error('Error fetching available movies:', error);
      toast.error('Failed to fetch available movies');
    } finally {
      setLoading(false);
    }
  };

  const searchMovieInTMDB = async (movieName) => {
    try {
      const response = await fetch(
        `${API_CONFIG.TMDB.API_URL}?api_key=${API_CONFIG.TMDB.API_KEY}&query=${encodeURIComponent(movieName)}&language=en-US&page=1`
      );
      
      if (!response.ok) {
        console.error('TMDB API error:', response.status, response.statusText);
        throw new Error('TMDB API request failed');
      }
      
      const data = await response.json();
      
      if (data.results && data.results.length > 0) {
        // Return the first result (most relevant match)
        return {
          ...data.results[0],
          original_name: movieName // Add the original name from our database
        };
      }
      
      console.log('No results found for:', movieName);
      return null; // No results found
    } catch (error) {
      console.error(`Error searching for movie "${movieName}":`, error);
      return null;
    }
  };

  // Filter movies based on search term
  const filteredMovies = movieDetails.filter(movie => 
    movie.title?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    movie.original_title?.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleMovieClick = (movie) => {
    setSelectedMovie(movie);
    setShowEditModal(true);
  };

  const handleEditWithAIClick = () => {
    setShowEditModal(false);
    setShowConfirmModal(true);
  };

  const handleConfirmEdit = () => {
    setShowConfirmModal(false);
    setShowEmailModal(true);
  };

  const handleEmailSubmit = async (e) => {
    e.preventDefault();
    
    if (!email || !email.includes('@')) {
      toast.error('Please enter a valid email address');
      return;
    }
    
    if (!selectedMovie) return;
    
    setEditLoading(true);
    toast.info('Starting movie processing. This may take a few minutes...', { autoClose: false });
    
    try {
      // Find the original movie file from availableMovies
      // Use the original_name property we added back
      const originalMovieFile = availableMovies.find(m => m.name === selectedMovie.original_name);
      
      if (!originalMovieFile) {
        console.error('Original movie file not found for:', selectedMovie.original_name);
        console.log('Available movies:', availableMovies);
        toast.error('Original movie file not found');
        setEditLoading(false);
        return;
      }
      
      console.log('Selected movie:', selectedMovie);
      console.log('Original movie file:', originalMovieFile);
      
      // Prepare the request payload
      const requestPayload = {
        title: originalMovieFile.name,
        blob_path: originalMovieFile.blob_path,
        youtube_url: null, // Empty string instead of null
        email: email
      };
      
      console.log('Request payload:', requestPayload);
      
      // Call the backend API to process the movie with AI
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.MOVIE_CLICKED}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestPayload),
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Server error:', errorData);
        console.error('Response status:', response.status);
        console.error('Response status text:', response.statusText);
        throw new Error(`Failed to process movie with AI: ${response.status} ${response.statusText}`);
      }
      
      const responseData = await response.json();
      console.log('Success response:', responseData);
      
      // Close all modals and reset the form
      setShowEmailModal(false);
      setShowConfirmModal(false);
      setShowEditModal(false);
      setEmail('');
      
      // Show success message
      toast.dismiss(); // Dismiss the loading toast
      toast.success('Movie processing started! Check your email for the result.', { autoClose: 5000 });
      
      // Reset the selected movie after a short delay
      setTimeout(() => {
        setSelectedMovie(null);
      }, 1000);
      
    } catch (error) {
      console.error('Error processing movie with AI:', error);
      toast.dismiss(); // Dismiss the loading toast
      toast.error(error.message || 'Failed to process movie with AI');
    } finally {
      setEditLoading(false);
    }
  };

  // Add a function to handle canceling the process
  const handleCancelProcess = () => {
    setShowEmailModal(false);
    setShowConfirmModal(false);
    setShowEditModal(false);
    setEmail('');
    setEditLoading(false);
    toast.dismiss(); // Dismiss any existing toasts
  };

  return (
    <div className="available-movies">
      <div className="section-header">
        <h2>Movie Library</h2>
        <div className="search-container">
          <input
            type="text"
            placeholder="Search by title..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="search-input"
          />
          <button 
            onClick={fetchAvailableMovies} 
            disabled={loading}
            className="fetch-button"
          >
            {loading ? 'Loading...' : 'Load Available Movies'}
          </button>
        </div>
      </div>

      {movieDetails.length > 0 ? (
        <div className="movies-grid">
          {filteredMovies.map((movie) => (
            <div 
              key={movie.id} 
              className="movie-card"
              onClick={() => handleMovieClick(movie)}
            >
              <img 
                src={movie.poster_path 
                  ? `${API_CONFIG.TMDB.IMAGE_BASE_URL}${movie.poster_path}` 
                  : 'https://via.placeholder.com/500x750'} 
                alt={movie.title} 
              />
              <div className="movie-info">
                <h3>{movie.title}</h3>
                {movie.release_date && (
                  <p>{movie.release_date.split('-')[0]}</p>
                )}
                {movie.vote_average && (
                  <p className="rating">⭐ {movie.vote_average.toFixed(1)}/10</p>
                )}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="no-movies">
          {loading 
            ? 'Loading movie details...' 
            : searchTerm 
              ? 'No movies found matching your search.'
              : 'No movies available. Click the button above to load movies.'}
        </p>
      )}

      {/* Edit Modal */}
      {showEditModal && selectedMovie && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Edit with AI</h2>
            <p>Do you want to create an AI recap for <strong>{selectedMovie.title}</strong>?</p>
            <div className="modal-actions">
              <button 
                className="cancel-button" 
                onClick={() => setShowEditModal(false)}
                disabled={editLoading}
              >
                Cancel
              </button>
              <button 
                className="edit-button" 
                onClick={handleEditWithAIClick}
                disabled={editLoading}
              >
                Edit with AI
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal */}
      {showConfirmModal && selectedMovie && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Confirm AI Processing</h2>
            <p>Are you sure you want to process <strong>{selectedMovie.title}</strong> with AI?</p>
            <p className="modal-note">This will create a recap video and send it to your email when complete.</p>
            <div className="modal-actions">
              <button 
                className="cancel-button" 
                onClick={() => setShowConfirmModal(false)}
                disabled={editLoading}
              >
                Cancel
              </button>
              <button 
                className="edit-button" 
                onClick={handleConfirmEdit}
                disabled={editLoading}
              >
                Confirm & Process
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Email Modal */}
      {showEmailModal && selectedMovie && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Enter Your Email</h2>
            <p>Please enter your email address to receive the AI recap for <strong>{selectedMovie.title}</strong>.</p>
            <form onSubmit={handleEmailSubmit}>
              <div className="email-input-container">
                <input
                  type="email"
                  placeholder="your.email@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="email-input"
                  required
                  disabled={editLoading}
                />
              </div>
              <div className="modal-actions">
                <button 
                  type="button"
                  className="cancel-button" 
                  onClick={handleCancelProcess}
                  disabled={editLoading}
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  className="edit-button" 
                  disabled={editLoading}
                >
                  {editLoading ? 'Processing...' : 'Submit & Process'}
                </button>
              </div>
            </form>
            {editLoading && (
              <div className="processing-indicator">
                <p>Processing your movie. This may take a few minutes...</p>
                <p>You'll receive an email when it's complete.</p>
                <button 
                  className="cancel-button" 
                  onClick={handleCancelProcess}
                  style={{ marginTop: '1rem' }}
                >
                  Return to Main Page
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default AvailableMovies; 