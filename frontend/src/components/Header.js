import React from 'react';
import './Header.css';

function Header({ isHumanDetected, connected }) {
  return (
    <header className="header">
      <div className="header-content">
        <h1>PERSON DETECTION DASHBOARD</h1>
        <p className="subtitle">Pretrained YOLO person detector</p>
      </div>

      <div className="status-indicator">
        <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
          <div className={`dot ${connected ? 'connected' : 'disconnected'}`}></div>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>

        <div className={`alert-status ${isHumanDetected ? 'danger' : 'safe'}`}>
          <div className={`status-dot ${isHumanDetected ? 'danger' : 'safe'}`}></div>
          <span>{isHumanDetected ? 'PERSON DETECTED' : 'No people detected'}</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
