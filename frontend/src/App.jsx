import React from 'react';
import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';
import MovieSearch from './components/MovieSearch';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Movie Search</h1>
      </header>
      <main>
        <MovieSearch />
      </main>
      <ToastContainer />
    </div>
  );
}

export default App; 