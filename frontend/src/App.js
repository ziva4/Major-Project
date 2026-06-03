import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './App.css';
import Header from './components/Header';
import VideoFeed from './components/VideoFeed';
import ControlPanel from './components/ControlPanel';
import AlertsPanel from './components/AlertsPanel';

function App() {
  const [socket, setSocket] = useState(null);
  const [detectionData, setDetectionData] = useState({
    confidence: 0,
    is_violence_detected: false,
    confidence_threshold: 0.6,
    frame_count: 0,
    alerts_count: 0
  });
  const [alerts, setAlerts] = useState([]);
  const [connected, setConnected] = useState(false);
  const [threshold, setThreshold] = useState(60);

  // Connect to backend
  useEffect(() => {
    const newSocket = io('http://localhost:5000', {
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5
    });

    newSocket.on('connect', () => {
      console.log('Connected to server');
      setConnected(true);
    });

    newSocket.on('disconnect', () => {
      console.log('Disconnected from server');
      setConnected(false);
    });

    newSocket.on('detection_update', (data) => {
      setDetectionData(data);
    });

    setSocket(newSocket);

    return () => newSocket.close();
  }, []);

  // Fetch alerts periodically
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch('http://localhost:5000/api/alerts?limit=10');
        const data = await response.json();
        setAlerts(data.alerts);
      } catch (error) {
        console.error('Error fetching alerts:', error);
      }
    };

    const interval = setInterval(fetchAlerts, 2000);
    fetchAlerts(); // Initial fetch

    return () => clearInterval(interval);
  }, []);

  const handleThresholdChange = (value) => {
    setThreshold(value);
    const thresholdPercent = value / 100;

    fetch('http://localhost:5000/api/threshold', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold: thresholdPercent })
    });
  };

  const handleAcknowledge = (alertId) => {
    fetch(`http://localhost:5000/api/alerts/${alertId}/acknowledge`, {
      method: 'POST'
    });
  };

  return (
    <div className="App">
      <Header
        isViolenceDetected={detectionData.is_violence_detected}
        connected={connected}
      />

      <div className="main-content">
        <div className="video-section">
          <VideoFeed
            confidence={detectionData.confidence}
            isViolenceDetected={detectionData.is_violence_detected}
          />
        </div>

        <div className="side-panel">
          <ControlPanel
            threshold={threshold}
            frameCount={detectionData.frame_count}
            alertCount={detectionData.alerts_count}
            onThresholdChange={handleThresholdChange}
          />
        </div>
      </div>

      <AlertsPanel
        alerts={alerts}
        onAcknowledge={handleAcknowledge}
      />
    </div>
  );
}

export default App;
