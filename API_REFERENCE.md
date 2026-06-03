# API Reference & Demo Scenarios

## REST API Endpoints

### Dashboard & Streaming

#### GET `/`
Serves the main dashboard HTML interface.
```
Response: HTML (text/html)
Status: 200 OK
```

#### GET `/video_feed`
Streams live video with confidence overlay as MJPEG.
```
Response: Multipart MJPEG stream (video/x-mixed-replace)
Status: 200 OK

Headers:
  Content-Type: multipart/x-mixed-replace; boundary=frame
  
Stream Format:
  --frame\r\n
  Content-Type: image/jpeg\r\n\r\n
  [JPEG data]\r\n
```

**Usage in HTML:**
```html
<img src="/video_feed" alt="Live Feed">
```

---

### Detection Data

#### GET `/api/detection`
Get current detection data including confidence score and alert status.
```
Method: GET
Response: application/json
Status: 200 OK

Response Body:
{
  "confidence": 0.4235,
  "is_violence_detected": false,
  "confidence_threshold": 0.6,
  "frame_count": 1024,
  "alerts_count": 3
}

Fields:
  confidence (float): Current confidence score [0.0-1.0]
  is_violence_detected (bool): Whether violence threshold exceeded
  confidence_threshold (float): Current detection threshold [0.0-1.0]
  frame_count (int): Total frames processed
  alerts_count (int): Total alerts triggered
```

**Example Usage:**
```bash
curl http://localhost:5000/api/detection

# Response
{
  "confidence": 0.4235,
  "is_violence_detected": false,
  "confidence_threshold": 0.6,
  "frame_count": 1024,
  "alerts_count": 3
}
```

---

### Alerts Management

#### GET `/api/alerts`
Retrieve recent alerts from the system.
```
Method: GET
Query Parameters:
  limit (int, optional): Number of alerts to return [1-100, default=10]

Response: application/json
Status: 200 OK

Response Body:
{
  "alerts": [
    {
      "id": 0,
      "timestamp": 1675000123.456,
      "confidence": 0.87,
      "frame_number": 512,
      "status": "ACTIVE"
    },
    {
      "id": 1,
      "timestamp": 1675000234.567,
      "confidence": 0.92,
      "frame_number": 1024,
      "status": "ACKNOWLEDGED"
    }
  ],
  "total": 2
}

Alert Fields:
  id (int): Unique alert identifier
  timestamp (float): Unix timestamp of detection
  confidence (float): Confidence score [0.0-1.0]
  frame_number (int): Frame where violence was detected
  status (string): "ACTIVE" or "ACKNOWLEDGED"
```

**Example Usage:**
```bash
# Get last 5 alerts
curl "http://localhost:5000/api/alerts?limit=5"

# Get last 20 alerts
curl "http://localhost:5000/api/alerts?limit=20"
```

---

#### POST `/api/alerts/{alert_id}/acknowledge`
Mark an alert as acknowledged by the operator.
```
Method: POST
URL Parameter:
  alert_id (int): ID of alert to acknowledge

Response: application/json
Status: 200 OK

Response Body:
{
  "success": true
}

Side Effects:
  - Sets alert status to "ACKNOWLEDGED"
  - Alert remains in history but marked as reviewed
```

**Example Usage:**
```bash
curl -X POST http://localhost:5000/api/alerts/0/acknowledge

# Response
{ "success": true }
```

---

### Settings & Configuration

#### POST `/api/threshold`
Set the confidence threshold for violence detection.
```
Method: POST
Content-Type: application/json
Body:
{
  "threshold": 0.65
}

Response: application/json
Status: 200 OK

Response Body:
{
  "success": true,
  "threshold": 0.65
}

Constraints:
  threshold (float): Must be between 0.0 and 1.0
  threshold < 0: Clamped to 0.0
  threshold > 1: Clamped to 1.0

Side Effects:
  - Immediately changes detection sensitivity
  - Lower values = more alerts (more sensitive)
  - Higher values = fewer alerts (less sensitive)
```

**Example Usage:**
```bash
# Set threshold to 50% (more sensitive)
curl -X POST http://localhost:5000/api/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.5}'

# Set threshold to 75% (less sensitive)
curl -X POST http://localhost:5000/api/threshold \
  -H "Content-Type: application/json" \
  -d '{"threshold": 0.75}'
```

---

## WebSocket Events

### Connection Events

#### `connect`
Client successfully connected to server.
```
Event: "connect"
Data: { "status": "Connected to server" }
Direction: Server → Client
```

#### `disconnect`
Client disconnected from server.
```
Event: "disconnect"
Direction: Server → Client
```

---

### Data Events

#### `detection_update`
Real-time detection data broadcast to all connected clients.
```
Event: "detection_update"
Direction: Server → Clients (broadcast every 0.5 seconds)

Data:
{
  "confidence": 0.5432,
  "is_violence_detected": true,
  "confidence_threshold": 0.6,
  "frame_count": 2048,
  "alerts_count": 5
}

Frequency: Every 500ms
Scope: All connected clients
```

#### `request_update`
Request immediate detection update from server.
```
Event: "request_update"
Direction: Client → Server

Response:
  Immediate "detection_update" event with current data
```

**JavaScript Example:**
```javascript
const socket = io();

// Automatic updates every 500ms
socket.on('detection_update', (data) => {
  console.log(`Confidence: ${(data.confidence*100).toFixed(1)}%`);
  console.log(`Violence: ${data.is_violence_detected}`);
});

// Manual update request
socket.emit('request_update');
```

---

## Demo Scenarios

### Scenario 1: Safe Environment Baseline

**Objective**: Establish baseline normal activity detection

**Steps**:
1. Start system: `python app.py`
2. Open dashboard: http://localhost:5000
3. Set confidence threshold to 60%
4. Sit still for 2 minutes

**Expected Results**:
- Confidence stays 0-20%
- Status remains "Safe" (green)
- No alerts generated
- Video feed shows normal processing

**Duration**: 5 minutes

---

### Scenario 2: High Motion (False Positive Test)

**Objective**: Test false positive rate with high-motion normal activity

**Steps**:
1. Start from Scenario 1
2. Move around camera normally
3. Wave hands randomly
4. Do normal exercising motions
5. Observe for 2 minutes

**Expected Results**:
- Confidence fluctuates 30-50%
- Occasional spikes above 60% but should smooth out
- Minimal false alerts
- Status mostly stays green

**Duration**: 5 minutes

**Observation**:
- If too many alerts: Increase threshold to 70%
- If no confidence increase: Check camera lighting

---

### Scenario 3: Simulated Threat Detection

**Objective**: Test system response to simulated violence

**Steps**:
1. Start from Scenario 1
2. Make rapid, aggressive arm movements
3. Simulate striking motions (toward air, not camera)
4. Make sudden jerky movements
5. Repeat for 30 seconds

**Expected Results**:
- Confidence jumps to 70-95%
- Status turns "VIOLENCE DETECTED" (red)
- Alert generated with high confidence
- Alert appears in alerts list
- Status indicator pulses

**Duration**: 2 minutes

---

### Scenario 4: Alert Management

**Objective**: Test alert acknowledgment and history

**Steps**:
1. Trigger alert (Scenario 3)
2. Observe alert in dashboard
3. Click "ACKNOWLEDGE" button
4. Verify status changes to "ACKNOWLEDGED"
5. Trigger multiple alerts (3-5)
6. Acknowledge some, leave others

**Expected Results**:
- Acknowledged alerts show different color/style
- Alert history maintains all records
- Can scroll through alert list
- Statistics show total alerts count

**Duration**: 5 minutes

---

### Scenario 5: Threshold Sensitivity Testing

**Objective**: Find optimal threshold for environment

**Steps**:
1. Set threshold to 40% (most sensitive)
2. Do normal activity for 2 minutes
3. Count false alerts
4. Increase threshold by 5%
5. Repeat for thresholds: 40, 45, 50, 55, 60, 65, 70%

**Expected Results**:
- Lower thresholds = more alerts
- Higher thresholds = fewer alerts
- Find sweet spot with minimal false positives
- Optimal typically 55-65% for mixed environments

**Duration**: 15 minutes

**Notes**:
```
Threshold 40%: May alert on normal fast motion
Threshold 50%: Balanced, occasional false alerts
Threshold 60%: Default, good balance
Threshold 70%: Higher false negatives
Threshold 80%: Very permissive, may miss violence
```

---

### Scenario 6: Multi-Alert Rapid Response

**Objective**: Test system under continuous threat

**Steps**:
1. Trigger alerts rapidly (Scenario 3, repeated)
2. Generate 5+ alerts in quick succession
3. Observe dashboard responsiveness
4. Check frame count and alert count

**Expected Results**:
- Dashboard remains responsive
- Alerts queue properly
- Frame processing continues at 30+ FPS
- No crashes or lag

**Duration**: 5 minutes

**Performance Metrics to Note**:
- Alert generation latency
- Dashboard update frequency
- CPU usage
- Memory consumption

---

### Scenario 7: Long-Duration Monitoring

**Objective**: Test stability over extended period

**Steps**:
1. Start system monitoring
2. Leave running for 30 minutes+
3. Observe for resource leaks
4. Check alert accumulation
5. Monitor dashboard performance

**Expected Results**:
- Stable CPU/Memory usage
- Smooth video stream
- No disconnections
- Dashboard responsive throughout
- Alert history properly maintained

**Duration**: 30+ minutes

---

### Scenario 8: Camera Failure Recovery

**Objective**: Test graceful handling of camera loss

**Steps**:
1. Start normal monitoring
2. Disconnect camera during operation
3. Observe dashboard behavior
4. Reconnect camera
5. Verify recovery

**Expected Results**:
- Video stream shows disconnection
- Error message or frozen frame
- Dashboard remains accessible
- No crashes
- Resumes monitoring after reconnection

**Duration**: 5 minutes

---

## Performance Benchmarks

### Expected Metrics

| Metric | Expected Value | Notes |
|--------|---|---|
| FPS | 25-35 | Depends on resolution & hardware |
| Latency (detection) | 100-150ms | Includes smoothing |
| CPU Usage | 15-25% | Single core utilization |
| Memory | 400-600MB | Including model + buffer |
| Model Size | 2.5MB | On disk |
| Alert Generation | <50ms | From detection to UI |
| WebSocket Latency | <10ms | Local network |

### Hardware Requirements

**Minimum:**
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 4GB
- Disk: 2GB free
- Camera: USB 2.0+

**Recommended:**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 8GB+
- Disk: 4GB free
- Camera: USB 3.0+
- GPU: NVIDIA GTX 1050+ (optional)

---

## Troubleshooting During Demo

### Issue: No frames showing
**Solution**:
- Check camera permission in Windows/Mac Settings
- Try different camera index
- Ensure no other app using camera

### Issue: High false alerts
**Solution**:
- Increase threshold to 70%
- Ensure good lighting
- Reduce background motion

### Issue: Slow dashboard
**Solution**:
- Close other applications
- Reduce video resolution in config
- Use wired network instead of WiFi

### Issue: Crashes
**Solution**:
- Check console for error messages
- Verify all dependencies installed
- Try `python backend/test_system.py`

---

## Quick Command Reference

```bash
# Start system
python app.py

# Test system (no Flask server)
python test_system.py

# Verify setup
python verify_setup.py

# Train model only
python train_model.py

# View config
python -c "from config import Config; print(Config())"
```

---

For more information, see:
- README_SYSTEM.md - Full system documentation
- ML_MODEL_DETAILS.md - Model architecture details
- QUICKSTART.md - Quick setup guide
