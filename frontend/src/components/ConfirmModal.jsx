import React, { useState } from 'react';
import './ConfirmModal.css';

const ConfirmModal = ({ isOpen, onConfirm, onCancel, movieTitle }) => {
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [email, setEmail] = useState('');
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [emailError, setEmailError] = useState('');
  const [youtubeError, setYoutubeError] = useState('');

  if (!isOpen) return null;

  const handleInitialConfirm = () => {
    setShowEmailInput(true);
  };

  const handleSubmit = () => {
    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setEmailError('Please enter a valid email address');
      return;
    }

    // YouTube URL validation
    const youtubeRegex = /^(https?:\/\/)?(www\.)?(youtube\.com|youtu\.be)\/.+$/;
    if (!youtubeRegex.test(youtubeUrl)) {
      setYoutubeError('Please enter a valid YouTube URL');
      return;
    }

    onConfirm(email, youtubeUrl);
    setShowEmailInput(false);
    setEmail('');
    setYoutubeUrl('');
    setEmailError('');
    setYoutubeError('');
  };

  const handleClose = () => {
    setShowEmailInput(false);
    setEmail('');
    setYoutubeUrl('');
    setEmailError('');
    setYoutubeError('');
    onCancel();
  };

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        {!showEmailInput ? (
          <>
            <h3>AI Movie Editor</h3>
            <p>Would you like AI to edit "{movieTitle}"?</p>
            <div className="modal-buttons">
              <button className="confirm-button" onClick={handleInitialConfirm}>
                Yes, Edit with AI
              </button>
              <button className="cancel-button" onClick={handleClose}>
                Cancel
              </button>
            </div>
          </>
        ) : (
          <>
            <h3>Enter Your Details</h3>
            <p>We'll notify you when the AI editing is complete</p>
            <div className="input-container">
              <input
                type="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setEmailError('');
                }}
                placeholder="Enter your email address"
                className="form-input"
              />
              {emailError && <p className="error-message">{emailError}</p>}
              
              <input
                type="url"
                value={youtubeUrl}
                onChange={(e) => {
                  setYoutubeUrl(e.target.value);
                  setYoutubeError('');
                }}
                placeholder="Enter YouTube URL of the movie"
                className="form-input"
              />
              {youtubeError && <p className="error-message">{youtubeError}</p>}
            </div>
            <div className="modal-buttons">
              <button 
                className="confirm-button" 
                onClick={handleSubmit}
              >
                Submit
              </button>
              <button className="cancel-button" onClick={handleClose}>
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ConfirmModal; 