import React, { useState } from 'react';
import { toast } from 'react-toastify';
import './MovieCard.css';
import ConfirmModal from './ConfirmModal';

const MovieCard = ({ movie }) => {
  const [showModal, setShowModal] = useState(false);

  const posterPath = movie.poster_path
    ? `https://image.tmdb.org/t/p/w500${movie.poster_path}`
    : 'https://via.placeholder.com/500x750';

  const handleMovieClick = () => {
    setShowModal(true);
  };

  const handleConfirm = async () => {
    setShowModal(false);
    
    // Show the AI editing notification
    toast.info('AI is editing your movie...', {
      position: "top-center",
      autoClose: 3000,
      hideProgressBar: false,
      closeOnClick: false,
      pauseOnHover: true,
      draggable: false,
    });

    try {
      const response = await fetch('http://localhost:8000/api/movie-clicked', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: movie.title
        }),
      });
      
      const data = await response.json();
      console.log('Backend response:', data);
      
    } catch (error) {
      console.error('Error sending movie click:', error);
      toast.error('Error processing your request');
    }
  };

  const handleCancel = () => {
    setShowModal(false);
  };

  return (
    <>
      <div className="movie-card" onClick={handleMovieClick}>
        <img src={posterPath} alt={movie.title} />
        <div className="movie-info">
          <h3>{movie.title}</h3>
          <p>{movie.release_date?.split('-')[0]}</p>
          <p className="rating">⭐ {movie.vote_average}/10</p>
        </div>
      </div>

      <ConfirmModal 
        isOpen={showModal}
        onConfirm={handleConfirm}
        onCancel={handleCancel}
        movieTitle={movie.title}
      />
    </>
  );
};

export default MovieCard; 