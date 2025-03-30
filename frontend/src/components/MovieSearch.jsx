import React, { useState } from 'react';
import MovieCard from './MovieCard';
import './MovieSearch.css';

const MovieSearch = () => {
  const [query, setQuery] = useState('');
  const [movies, setMovies] = useState([]);
  const [loading, setLoading] = useState(false);

  // Replace with your actual API key
  const API_KEY = '0290374b307bbe7b264ac9b24ff98a72';
  const API_URL = 'https://api.themoviedb.org/3/search/movie';

  const searchMovies = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const response = await fetch(
        `${API_URL}?api_key=${API_KEY}&query=${query}&language=en-US&page=1`
      );
      const data = await response.json();
      setMovies(data.results);
    } catch (error) {
      console.error('Error fetching movies:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="movie-search">
      <form onSubmit={searchMovies}>
        <input
          type="text"
          placeholder="Search for a movie..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button type="submit">Search</button>
      </form>

      <div className="movies-grid">
        {loading ? (
          <p>Loading...</p>
        ) : (
          movies.map((movie) => <MovieCard key={movie.id} movie={movie} />)
        )}
      </div>
    </div>
  );
};

export default MovieSearch; 