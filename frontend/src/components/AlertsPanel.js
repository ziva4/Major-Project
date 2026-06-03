import React from 'react';
import './AlertsPanel.css';

function AlertsPanel({ alerts, onAcknowledge }) {
  const formatTime = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleTimeString();
  };

  return (
    <div className="alerts-section">
      <div className="alerts-title">
        <span className="alert-icon">🚨</span>
        RECENT ALERTS (Police Dispatch)
      </div>

      <div className="alerts-list">
        {alerts.length === 0 ? (
          <div className="no-alerts">
            <p>No alerts yet. System is monitoring...</p>
          </div>
        ) : (
          alerts.map((alert) => (
            <div
              key={alert.id}
              className={`alert-item ${alert.status === 'ACKNOWLEDGED' ? 'acknowledged' : ''}`}
            >
              <div className="alert-details">
                <div className="alert-time">{formatTime(alert.timestamp)}</div>
                <div className="alert-confidence">
                  Confidence: {(alert.confidence * 100).toFixed(1)}%
                </div>
                <div className="alert-frame">Frame: {alert.frame_number}</div>
              </div>

              <div className="alert-actions">
                <div className={`alert-status ${alert.status === 'ACKNOWLEDGED' ? 'acknowledged' : 'active'}`}>
                  {alert.status}
                </div>

                {alert.status !== 'ACKNOWLEDGED' && (
                  <button
                    className="ack-btn"
                    onClick={() => onAcknowledge(alert.id)}
                  >
                    ACKNOWLEDGE
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

export default AlertsPanel;
