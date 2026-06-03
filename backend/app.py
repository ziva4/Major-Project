from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit, disconnect
from flask_cors import CORS
import threading
import time
from video_processor import VideoProcessor

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize video processor
processor = VideoProcessor(confidence_threshold=0.6)

def emit_data():
    """Emit detection data to connected clients"""
    while True:
        try:
            data = processor.get_data()
            socketio.emit('detection_update', data, broadcast=True)
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in emit_data: {e}")
            time.sleep(1)

@app.route('/')
def index():
    """Serve main dashboard"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream video with detections"""
    def generate():
        while True:
            try:
                frame = processor.get_frame_jpeg()
                if frame:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                time.sleep(0.033)  # ~30 FPS
            except Exception as e:
                print(f"Error in video_feed: {e}")
                break
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detection', methods=['GET'])
def get_detection():
    """Get current detection data"""
    return jsonify(processor.get_data())

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        'alerts': processor.get_alerts(limit),
        'total': len(processor.alerts)
    })

@app.route('/api/threshold', methods=['POST'])
def set_threshold():
    """Set confidence threshold"""
    data = request.json
    threshold = data.get('threshold', 0.6)
    processor.set_threshold(threshold)
    return jsonify({'success': True, 'threshold': processor.confidence_threshold})

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    processor.acknowledge_alert(alert_id)
    return jsonify({'success': True})

@socketio.on('connect')
def handle_connect():
    """Handle WebSocket connection"""
    print(f"Client connected: {request.sid}")
    emit('connect_response', {'status': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle WebSocket disconnection"""
    print(f"Client disconnected: {request.sid}")

@socketio.on('request_update')
def handle_update_request():
    """Handle manual update request"""
    data = processor.get_data()
    emit('detection_update', data)

if __name__ == '__main__':
    print("🚨 Violence Detection System Starting...")

    # Start video processor
    try:
        processor.start()
        print("✓ Video processor started")
    except Exception as e:
        print(f"✗ Error starting video processor: {e}")
        exit(1)

    # Start emission thread
    emission_thread = threading.Thread(target=emit_data, daemon=True)
    emission_thread.start()

    # Start Flask app
    print("🚀 Starting Flask server on http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
