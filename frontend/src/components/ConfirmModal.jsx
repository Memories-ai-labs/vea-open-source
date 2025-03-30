import React, { useState } from 'react';
import './ConfirmModal.css';

const ConfirmModal = ({ isOpen, onConfirm, onCancel, movieTitle }) => {
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');

  if (!isOpen) return null;

  const handleInitialConfirm = () => {
    setShowEmailInput(true);
  };

  const handleEmailSubmit = () => {
    // Basic email validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      setEmailError('Please enter a valid email address');
      return;
    }
    onConfirm(email);
    setShowEmailInput(false);
    setEmail('');
    setEmailError('');
  };

  const handleClose = () => {
    setShowEmailInput(false);
    setEmail('');
    setEmailError('');
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
            <h3>Enter Your Email</h3>
            <p>We'll notify you when the AI editing is complete</p>
            <div className="email-input-container">
              <input
                type="email"
                value={email}
                onChange={(e) => {
                  setEmail(e.target.value);
                  setEmailError('');
                }}
                placeholder="Enter your email address"
                className="email-input"
              />
              {emailError && <p className="error-message">{emailError}</p>}
            </div>
            <div className="modal-buttons">
              <button 
                className="confirm-button" 
                onClick={handleEmailSubmit}
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