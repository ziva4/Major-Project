import React from 'react';
import './VideoFeed.css';

function VideoFeed({ confidence, isHumanDetected }) {
  return (
    <div className="video-panel">
      <div className="video-title">📹 Live Camera Feed</div>
      <div className="video-container">
        <img
          className="video-feed"
          src="http://localhost:5000/video_feed"
          alt="Live Video Stream"
        />
        <div className="video-overlay">
          <div className="confidence-display">
            <div className="confidence-label">Highest Person Confidence</div>
            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{ width: `${confidence * 100}%` }}
              >
                {(confidence * 100).toFixed(0)}%
              </div>
            </div>
            <div className="confidence-text">
              {(confidence * 100).toFixed(2)}%
            </div>
          </div>

          <div className="status-overlay">
            <div className={`status-badge ${isHumanDetected ? 'danger' : 'safe'}`}>
              {isHumanDetected ? 'PERSON DETECTED' : 'No people detected'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default VideoFeed;
