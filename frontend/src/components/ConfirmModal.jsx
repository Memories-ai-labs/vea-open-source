import React from 'react';
import './ConfirmModal.css';

const ConfirmModal = ({ isOpen, onConfirm, onCancel, movieTitle }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-content" onClick={e => e.stopPropagation()}>
        <h3>AI Movie Editor</h3>
        <p>Would you like AI to edit "{movieTitle}"?</p>
        <div className="modal-buttons">
          <button className="confirm-button" onClick={onConfirm}>
            Yes, Edit with AI
          </button>
          <button className="cancel-button" onClick={onCancel}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmModal; 