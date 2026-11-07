from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import threading
import time
import argparse
import torch
import uuid
from pathlib import Path
import cv2
from werkzeug.exceptions import RequestEntityTooLarge
from video_processor import VideoProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 256 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'data' / 'uploads'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Initialize video processor
processor = None
source_lock = threading.Lock()


@app.errorhandler(RequestEntityTooLarge)
def upload_too_large(error):
    return jsonify({'error': 'Choose a video smaller than 256 MB.'}), 413


@app.post('/api/source/live')
def start_live():
    if not source_lock.acquire(blocking=False):
        return jsonify({'error': 'A source change is already in progress.'}), 409
    try:
        processor.stop()
        processor.start()
        return jsonify({'success': True, 'source': processor.get_data()})
    except (RuntimeError, ValueError, OSError) as error:
        return jsonify({'error': str(error)}), 409
    finally:
        source_lock.release()


@app.post('/api/source/upload')
def upload_video():
    uploaded = request.files.get('video')
    if uploaded is None or not uploaded.filename:
        return jsonify({'error': 'Choose a video file first.'}), 400
    extension = Path(uploaded.filename).suffix.lower()
    if extension not in {'.mp4', '.avi', '.mov', '.mkv', '.webm'}:
        return jsonify({'error': 'Supported formats: MP4, AVI, MOV, MKV, WebM.'}), 415
    if not source_lock.acquire(blocking=False):
        return jsonify({'error': 'A source change is already in progress.'}), 409
    saved_path = None
    started = False
    try:
        folder = Path(app.config['UPLOAD_FOLDER'])
        folder.mkdir(parents=True, exist_ok=True)
        saved_path = folder / (uuid.uuid4().hex + extension)
        uploaded.save(saved_path)
        capture = cv2.VideoCapture(str(saved_path))
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            readable, frame = capture.read()
            if not capture.isOpened() or total < 16 or not readable or frame is None or not (0 < fps < float('inf')):
                return jsonify({'error': 'Video must be readable with at least 16 frames and valid FPS.'}), 422
        finally:
            capture.release()
        processor.stop()
        processor.start_video(saved_path, display_name=uploaded.filename.replace('\\', '/').split('/')[-1])
        started = True
        return jsonify({'success': True, 'source': processor.get_data()})
    except (RuntimeError, ValueError) as error:
        return jsonify({'error': str(error)}), 409
    except OSError:
        return jsonify({'error': 'Could not save the uploaded video.'}), 500
    finally:
        try:
            if not started and saved_path is not None and saved_path.exists():
                saved_path.unlink()
        finally:
            source_lock.release()


@app.post('/api/source/stop')
def stop_source():
    if not source_lock.acquire(blocking=False):
        return jsonify({'error': 'A source change is already in progress.'}), 409
    try:
        processor.stop()
        return jsonify({'success': True})
    finally:
        source_lock.release()


@app.post('/api/source/playback')
def playback_control():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or data.get('action') not in ('pause', 'resume'):
        return jsonify({'error': 'Choose pause or resume.'}), 400
    try:
        processor.set_paused(data['action'] == 'pause')
        return jsonify({'success': True})
    except ValueError as error:
        return jsonify({'error': str(error)}), 409


@app.before_request
def require_processor():
    if processor is None and request.path != '/':
        return jsonify({'error': 'Video processor has not been started.'}), 503

def emit_data():
    """Emit detection data to connected clients"""
    while True:
        try:
            data = processor.get_data()
            with app.app_context():
                socketio.emit('detection_update', data)
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
        'total': processor.get_data()['alerts_count']
    })

@app.route('/api/threshold', methods=['POST'])
def set_threshold():
    """Set minimum confidence required to show a detected person."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or isinstance(data.get('threshold'), bool):
        return jsonify({'error': 'Supply a numeric threshold in JSON.'}), 400
    try:
        processor.set_threshold(data['threshold'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Supply a finite numeric threshold.'}), 400
    return jsonify({'success': True, 'threshold': processor.confidence_threshold})

@app.route('/api/alerts/<int:alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    if not processor.acknowledge_alert(alert_id):
        return jsonify({'error': 'Alert not found.'}), 404
    return jsonify({'success': True})


@app.post('/api/alert-threshold')
def set_alert_threshold():
    """Set the UI's demo alert threshold and enable review alerts."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or isinstance(data.get('threshold'), bool):
        return jsonify({'error': 'Supply a demo alert threshold between 0 and 1.'}), 400
    try:
        processor.set_alert_threshold(data['threshold'])
    except (KeyError, TypeError, ValueError):
        return jsonify({'error': 'Supply a finite demo alert threshold between 0 and 1.'}), 400
    state = processor.get_data()
    return jsonify({'success': True, 'threshold': state['violence_threshold'],
                    'alerts_enabled': state['alerts_enabled']})

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
    if processor is not None:
        emit('detection_update', processor.get_data())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--demo-alerts', action='store_true')
    parser.add_argument('--alert-threshold', type=float, default=0.70)
    args = parser.parse_args()
    torch.set_num_threads(4)
    processor = VideoProcessor(camera_index=args.camera, alerts_enabled=args.demo_alerts,
                               alert_threshold=args.alert_threshold)
    print("Person Detection System Starting...")

    print("Choose Live camera or Upload video in the dashboard to start detection.")

    # Start emission thread
    emission_thread = threading.Thread(target=emit_data, daemon=True)
    emission_thread.start()

    # Start Flask app
    print("🚀 Starting Flask server on http://localhost:5000")
    try:
        socketio.run(app, host='127.0.0.1', port=5000, debug=False)
    finally:
        processor.stop()
