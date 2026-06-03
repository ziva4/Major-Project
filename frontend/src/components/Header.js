import React from 'react';
import './Header.css';

function Header({ isViolenceDetected, connected }) {
  return (
    <header className="header">
      <div className="header-content">
        <h1>🚨 VIOLENCE DETECTION DASHBOARD</h1>
        <p className="subtitle">Police Alert System</p>
      </div>

      <div className="status-indicator">
        <div className={`connection-status ${connected ? 'connected' : 'disconnected'}`}>
          <div className={`dot ${connected ? 'connected' : 'disconnected'}`}></div>
          <span>{connected ? 'Connected' : 'Disconnected'}</span>
        </div>

        <div className={`alert-status ${isViolenceDetected ? 'danger' : 'safe'}`}>
          <div className={`status-dot ${isViolenceDetected ? 'danger' : 'safe'}`}></div>
          <span>{isViolenceDetected ? '⚠️ VIOLENCE DETECTED!' : '✓ Safe'}</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
