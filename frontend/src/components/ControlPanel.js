import React from 'react';
import './ControlPanel.css';

function ControlPanel({
  threshold,
  frameCount,
  alertCount,
  onThresholdChange
}) {
  return (
    <div className="controls-panel">
      <div className="control-section">
        <h3>⚙️ SETTINGS</h3>
        <div className="threshold-control">
          <span className="threshold-label">Low</span>
          <input
            type="range"
            min="0"
            max="100"
            value={threshold}
            onChange={(e) => onThresholdChange(parseInt(e.target.value))}
            className="threshold-slider"
          />
          <span className="threshold-label">High</span>
        </div>
        <div className="threshold-value">{threshold}%</div>
      </div>

      <div className="control-section">
        <h3>📊 STATISTICS</h3>
        <div className="stats">
          <div className="stat-box">
            <div className="stat-label">Frames Processed</div>
            <div className="stat-value">{frameCount}</div>
          </div>
          <div className="stat-box">
            <div className="stat-label">People Detected</div>
            <div className="stat-value">{alertCount}</div>
          </div>
        </div>
      </div>

      <div className="control-section info-section">
        <h3>ℹ️ INFO</h3>
        <div className="info-text">
          <p><strong>System:</strong> Active</p>
          <p><strong>Model:</strong> YOLO11n Person Detector</p>
          <p><strong>FPS:</strong> 30</p>
          <p><strong>Resolution:</strong> 640×480</p>
        </div>
      </div>
    </div>
  );
}

export default ControlPanel;
