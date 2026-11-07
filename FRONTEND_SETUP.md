> **Current status ? 7 November 2026:** The built-in Flask dashboard now uses
> YOLO11n/ByteTrack and MobileNetV3 temporal pooling + logistic regression.
> Live camera and uploaded videos are supported. The demo-alert slider defaults
> to 70%; person detection remains at 50%. Logistic regression achieved 76.00%
> held-out test accuracy and 74.74% test F1. KNN has validation results only.
> Read [README.md](README.md) and [README_SYSTEM.md](README_SYSTEM.md) first.
> The older material below is historical and may describe superseded interfaces,
> dependencies or model implementations; it is not the current setup procedure.

# Frontend Setup & Usage Guide

## 🚀 Quick Start

### Option 1: Use Built-in Flask Dashboard (Recommended for Quick Start)

The system comes with a built-in HTML dashboard served by Flask. No npm/Node.js required:

```bash
cd backend
run.bat  # Windows
# or
./run.sh  # Linux/macOS
```

Then open: **http://localhost:5000**

---

### Option 2: Run React Frontend (Advanced)

For a modern React development environment with hot-reloading:

## Prerequisites

- **Node.js** 14+ - Download from https://nodejs.org
- **npm** (comes with Node.js)
- **Backend running** on http://localhost:5000

## Installation

### Windows

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

### Linux/macOS

```bash
cd frontend
npm install
npm start
```

Server will open at **http://localhost:3000**

## Frontend Structure

```
frontend/
├── public/
│   └── index.html              # HTML template
├── src/
│   ├── components/
│   │   ├── Header.js           # Top status bar
│   │   ├── VideoFeed.js        # Live video display
│   │   ├── ControlPanel.js     # Settings & stats
│   │   ├── AlertsPanel.js      # Alert management
│   │   └── *.css               # Component styles
│   ├── App.js                  # Main application
│   ├── App.css                 # Global styles
│   ├── index.js                # Entry point
│   └── index.css               # Base styles
└── package.json                # Dependencies
```

## Features

### Header Component
```
┌─────────────────────────────────────────────┐
│ 🚨 VIOLENCE DETECTION DASHBOARD             │
│ Status: Connected | Violence: Safe    🟢    │
└─────────────────────────────────────────────┘
```

### Video Feed
- Live MJPEG stream from backend camera
- Confidence bar (0-100%)
- Real-time status overlay
- Professional overlay graphics

### Control Panel
- Confidence threshold slider (0-100%)
- Statistics display:
  - Frames processed count
  - Total alerts count
  - System information

### Alerts Panel
- Recent alerts list (scrollable)
- Alert timestamp
- Confidence percentage
- Quick acknowledge button
- Status: ACTIVE / ACKNOWLEDGED

## Available Commands

```bash
# Start development server (with hot reload)
npm start

# Build for production
npm run build

# Run tests
npm test

# Eject (⚠️ not reversible)
npm run eject
```

## API Integration

The React frontend connects to Flask backend via:

### 1. REST Endpoints
```javascript
// Get detection data
fetch('http://localhost:5000/api/detection')

// Get recent alerts
fetch('http://localhost:5000/api/alerts?limit=10')

// Set confidence threshold
fetch('http://localhost:5000/api/threshold', {
  method: 'POST',
  body: JSON.stringify({ threshold: 0.65 })
})

// Acknowledge alert
fetch('http://localhost:5000/api/alerts/{id}/acknowledge', {
  method: 'POST'
})
```

### 2. WebSocket Events
```javascript
// Real-time updates (every 500ms)
socket.on('detection_update', (data) => {
  // {
  //   confidence: 0.5234,
  //   is_violence_detected: false,
  //   frame_count: 1024,
  //   alerts_count: 2
  // }
});

// Manual update request
socket.emit('request_update');
```

### 3. Video Streaming
```html
<!-- MJPEG stream -->
<img src="http://localhost:5000/video_feed" alt="Live Feed" />
```

## Configuration

To connect to different backend:

Edit **src/App.js**:
```javascript
const newSocket = io('http://YOUR_IP:PORT', {
  reconnection: true,
  // ... other options
});
```

Update all fetch URLs:
```javascript
fetch('http://YOUR_IP:PORT/api/...')
```

## CORS Configuration

If running frontend and backend on different servers, ensure Flask CORS is enabled (already configured in `backend/app.py`):

```python
from flask_cors import CORS
CORS(app)  # Already done in backend
```

## Development

### Hot Module Reloading

Changes to React components automatically reload in browser during development:

```bash
# Terminal 1: Start backend
cd backend
python app.py

# Terminal 2: Start frontend
cd frontend
npm start
```

### Debugging

**Browser DevTools** (Press F12):
- **Console** tab: View logs and errors
- **Network** tab: Monitor API calls and WebSocket
- **React DevTools**: Install Chrome extension for component debugging

```javascript
// Add logs to components
console.log('Component mounted');
console.log('Detection data:', detectionData);
```

### Add New Component

1. Create `frontend/src/components/MyComponent.js`:
```javascript
import React from 'react';
import './MyComponent.css';

function MyComponent() {
  return <div>My Component</div>;
}

export default MyComponent;
```

2. Create `frontend/src/components/MyComponent.css`:
```css
/* Your styles */
```

3. Use in `App.js`:
```javascript
import MyComponent from './components/MyComponent';

// In JSX
<MyComponent />
```

## Production Build

Create optimized production bundle:

```bash
npm run build
```

This creates a `build/` folder (~200KB gzipped) ready for deployment.

### Deploy to Vercel (Free)

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd frontend
vercel
```

### Deploy to Netlify (Free)

```bash
# Build first
npm run build

# Drag & drop 'build' folder to Netlify
# Or use Netlify CLI:
npm install -g netlify-cli
netlify deploy --prod --dir=build
```

### Self-Hosted Deployment

```bash
# Build
npm run build

# Copy to web server
scp -r build/* user@server:/var/www/html/

# Or with Docker
docker build -t violence-detection-frontend .
docker run -p 80:3000 violence-detection-frontend
```

## Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build
FROM node:18-alpine
RUN npm install -g serve
COPY --from=0 /app/build ./build
CMD ["serve", "-s", "build", "-l", "3000"]
EXPOSE 3000
```

Build and run:
```bash
docker build -t violence-detection-frontend .
docker run -p 3000:3000 violence-detection-frontend
```

## Troubleshooting

### npm install fails
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and package-lock.json
rm -rf node_modules package-lock.json

# Reinstall
npm install
```

### Port 3000 already in use
```bash
# Use different port
PORT=3001 npm start

# Or kill process using port 3000
# Windows: netstat -ano | findstr :3000
# Linux/Mac: lsof -i :3000
```

### Connection to backend fails
1. Verify backend is running: `http://localhost:5000`
2. Check backend logs for errors
3. Verify CORS is enabled
4. Check firewall settings
5. Try adding backend URL to hosts file

### Slow performance
```bash
# Use production build
npm run build
serve -s build
```

- Close other browser tabs
- Clear browser cache
- Disable browser extensions
- Check network latency

### Video not showing
1. Check camera permissions
2. Verify `/video_feed` endpoint works
3. Try accessing directly: http://localhost:5000/video_feed
4. Check backend console for errors

### WebSocket disconnects
1. Check backend is serving on port 5000
2. Verify Socket.IO is enabled
3. Check browser console for connection errors
4. Try connecting from different network

## Performance Optimization

### Bundle Size
```bash
npm run build

# Analyze bundle
npm install -g webpack-bundle-analyzer
# Build then inspect
```

### Network
- Enable gzip compression on server
- Use CDN for static assets
- Lazy load components for large apps
- Compress images

### Runtime
- Memoize expensive computations
- Use React.memo for components
- Virtualize long lists
- Debounce frequent operations

## Browser Support

| Browser | Support |
|---------|---------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |
| Mobile Chrome | 90+ |
| Mobile Safari | 14+ |

## Environment Variables

Create `.env`:
```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=http://localhost:5000
```

Use in code:
```javascript
const apiUrl = process.env.REACT_APP_API_URL;
```

## Testing

```bash
# Run tests
npm test

# Run with coverage
npm test -- --coverage

# Run specific test
npm test -- VideoFeed.test.js
```

Example test:
```javascript
import { render, screen } from '@testing-library/react';
import Header from './Header';

test('renders header title', () => {
  render(<Header isViolenceDetected={false} connected={true} />);
  expect(screen.getByText(/VIOLENCE DETECTION/i)).toBeInTheDocument();
});
```

## Documentation

- **React Docs**: https://react.dev
- **Socket.IO Docs**: https://socket.io/docs/
- **Material-UI**: https://mui.com (optional UI library)
- **Axios**: https://axios-http.com (optional HTTP client)

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Blank page | Check console (F12) for errors |
| Connection refused | Backend not running on 5000 |
| Video not loading | Camera permissions or backend issue |
| Slow performance | Close tabs, use production build |
| Port already in use | Use different port with PORT=3001 npm start |

## Best Practices

1. ✓ Use production build for deployment
2. ✓ Enable HTTPS in production
3. ✓ Implement authentication/login
4. ✓ Add error boundaries
5. ✓ Log important events
6. ✓ Monitor performance
7. ✓ Regular backups
8. ✓ Security updates

---

**Need help?**
- Check backend logs: `python app.py`
- Check browser console: F12 → Console tab
- Review troubleshooting section above
- See main README_SYSTEM.md for architecture details
