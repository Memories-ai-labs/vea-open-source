import React from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import MovieSearch from './components/MovieSearch';
import AvailableMovies from './components/AvailableMovies';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Search</h1>
      </header>
      <main>
        {/* <MovieSearch /> */}
        <AvailableMovies />
      </main>
      <ToastContainer />
    </div>
  );
}

export default App; 