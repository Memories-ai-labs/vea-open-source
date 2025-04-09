import React, { useState } from 'react';
import { toast } from 'react-toastify';
import './MovieCard.css';
import API_CONFIG from '../config';

const MovieCard = ({ movie }) => {
  const [showEditModal, setShowEditModal] = useState(false);
  const [showConfirmModal, setShowConfirmModal] = useState(false);
  const [showEmailModal, setShowEmailModal] = useState(false);
  const [email, setEmail] = useState('');
  const [editLoading, setEditLoading] = useState(false);

  const handleEditWithAI = () => {
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
    
    setEditLoading(true);
    try {
      const response = await fetch(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.MOVIE_CLICKED}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: movie.title,
          blob_path: movie.blob_path,
          youtube_url: movie.youtube_url,
          email: email
        }),
      });
      
      if (!response.ok) {
        throw new Error('Failed to process movie with AI');
      }
      
      await response.json();
      toast.success('Movie processing started! Check your email for the result.');
      setShowEmailModal(false);
      setEmail('');
    } catch (error) {
      console.error('Error processing movie with AI:', error);
      toast.error('Failed to process movie with AI');
    } finally {
      setEditLoading(false);
    }
  };

  return (
    <div className="movie-card">
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
        <button 
          className="edit-button" 
          onClick={() => setShowEditModal(true)}
        >
          Edit with AI
        </button>
      </div>

      {/* Edit Modal */}
      {showEditModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Edit with AI</h2>
            <p>Do you want to create an AI recap for <strong>{movie.title}</strong>?</p>
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
                onClick={handleEditWithAI}
                disabled={editLoading}
              >
                Edit with AI
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Confirmation Modal */}
      {showConfirmModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Confirm AI Processing</h2>
            <p>Are you sure you want to process <strong>{movie.title}</strong> with AI?</p>
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
      {showEmailModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2>Enter Your Email</h2>
            <p>Please enter your email address to receive the AI recap for <strong>{movie.title}</strong>.</p>
            <form onSubmit={handleEmailSubmit}>
              <div className="email-input-container">
                <input
                  type="email"
                  placeholder="your.email@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="email-input"
                  required
                />
              </div>
              <div className="modal-actions">
                <button 
                  type="button"
                  className="cancel-button" 
                  onClick={() => {
                    setShowEmailModal(false);
                    setEmail('');
                  }}
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
          </div>
        </div>
      )}
    </div>
  );
};

export default MovieCard; 