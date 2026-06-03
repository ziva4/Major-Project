#!/usr/bin/env python3
"""
Setup Verification Script - Violence Detection System
Checks all dependencies and system requirements
"""

import sys
import subprocess
import importlib.util
import cv2
import os
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_header(text):
    print(f"\n{Colors.BLUE}{'='*50}")
    print(f"{text}")
    print(f"{'='*50}{Colors.END}\n")

def check_mark():
    return f"{Colors.GREEN}✓{Colors.END}"

def cross_mark():
    return f"{Colors.RED}✗{Colors.END}"

def warning_mark():
    return f"{Colors.YELLOW}⚠{Colors.END}"

def check_python_version():
    """Check Python version"""
    print_header("Python Environment")

    version = sys.version_info
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        print(f"{check_mark()} Python 3.8+ required - PASS")
        return True
    else:
        print(f"{cross_mark()} Python 3.8+ required - FAIL")
        return False

def check_dependencies():
    """Check required Python packages"""
    print_header("Python Dependencies")

    required = {
        'flask': 'Flask',
        'flask_socketio': 'Flask-SocketIO',
        'flask_cors': 'Flask-CORS',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'tensorflow': 'TensorFlow',
        'sklearn': 'scikit-learn',
        'PIL': 'Pillow'
    }

    all_ok = True
    for module, name in required.items():
        spec = importlib.util.find_spec(module)
        if spec is not None:
            try:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"{check_mark()} {name:<20} {version}")
            except Exception as e:
                print(f"{warning_mark()} {name:<20} (import error)")
                all_ok = False
        else:
            print(f"{cross_mark()} {name:<20} NOT INSTALLED")
            all_ok = False

    return all_ok

def check_camera():
    """Check if camera is available"""
    print_header("Camera Detection")

    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()

        if ret:
            h, w = frame.shape[:2]
            print(f"{check_mark()} Camera Found")
            print(f"    Resolution: {w}x{h}")
            print(f"    Frame: ✓ Captured successfully")
            return True
        else:
            print(f"{warning_mark()} Camera Detected but cannot read frames")
            return False
    else:
        print(f"{cross_mark()} Camera NOT FOUND")
        print("    Please check:")
        print("    • Camera is connected")
        print("    • Camera is enabled in system settings")
        print("    • No other app is using the camera")
        return False

def check_disk_space():
    """Check available disk space"""
    print_header("Disk Space")

    try:
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)

        print(f"Total: {total_gb:.1f} GB")
        print(f"Free: {free_gb:.1f} GB")

        if free_gb >= 2:
            print(f"{check_mark()} Sufficient space for models and data")
            return True
        else:
            print(f"{warning_mark()} Less than 2GB free space")
            return False
    except Exception as e:
        print(f"{warning_mark()} Could not check disk space: {e}")
        return False

def check_files():
    """Check important project files"""
    print_header("Project Files")

    files_to_check = {
        'app.py': 'Main Flask app',
        'video_processor.py': 'Video processing module',
        'train_model.py': 'Model training script',
        'templates/index.html': 'Dashboard HTML',
        'config.json': 'Configuration file'
    }

    all_ok = True
    for filename, description in files_to_check.items():
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"{check_mark()} {filename:<25} ({size} bytes) - {description}")
        else:
            print(f"{cross_mark()} {filename:<25} NOT FOUND")
            all_ok = False

    # Check for models directory
    if os.path.exists('models'):
        print(f"{check_mark()} models/ directory exists")
        model_file = 'models/violence_detector.h5'
        if os.path.exists(model_file):
            size = os.path.getsize(model_file) / (1024**2)
            print(f"    {check_mark()} Pre-trained model found ({size:.1f} MB)")
        else:
            print(f"    {warning_mark()} No pre-trained model (will train on first run)")
    else:
        print(f"{warning_mark()} models/ directory will be created on first run")

    return all_ok

def check_port():
    """Check if port 5000 is available"""
    print_header("Network Port")

    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        result = sock.connect_ex(('localhost', 5000))
        if result != 0:
            print(f"{check_mark()} Port 5000 is available")
            return True
        else:
            print(f"{warning_mark()} Port 5000 is already in use")
            print("    Solution: Change port in app.py line 105")
            return False
    except Exception as e:
        print(f"{warning_mark()} Could not check port: {e}")
        return False
    finally:
        sock.close()

def check_memory():
    """Check available system memory"""
    print_header("System Memory")

    try:
        import psutil
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        total_gb = memory.total / (1024**3)
        percent = memory.percent

        print(f"Total: {total_gb:.1f} GB")
        print(f"Available: {available_gb:.1f} GB ({100-percent:.0f}%)")

        if available_gb >= 1:
            print(f"{check_mark()} Sufficient memory available")
            return True
        else:
            print(f"{warning_mark()} Low available memory (<1GB)")
            return False
    except ImportError:
        print(f"{warning_mark()} psutil not installed (optional)")
        return True

def generate_report(results):
    """Generate and display verification report"""
    print_header("Verification Report")

    passed = sum(results.values())
    total = len(results)
    percentage = (passed / total) * 100

    for check, result in results.items():
        status = check_mark() if result else cross_mark()
        print(f"{status} {check}")

    print(f"\n{Colors.BLUE}Overall: {passed}/{total} checks passed ({percentage:.0f}%){Colors.END}")

    if percentage == 100:
        print(f"\n{Colors.GREEN}✓ System is ready! Run 'python app.py' to start{Colors.END}\n")
    elif percentage >= 80:
        print(f"\n{Colors.YELLOW}⚠ System is mostly ready. Check warnings above{Colors.END}\n")
    else:
        print(f"\n{Colors.RED}✗ System has issues. Fix errors before running{Colors.END}\n")

    return percentage == 100

def main():
    """Run all checks"""
    print(f"\n{Colors.BLUE}{'='*50}")
    print("Violence Detection System - Setup Verification")
    print(f"{'='*50}{Colors.END}")

    results = {
        'Python Version': check_python_version(),
        'Dependencies': check_dependencies(),
        'Camera Detection': check_camera(),
        'Disk Space': check_disk_space(),
        'Project Files': check_files(),
        'Network Port': check_port(),
        'System Memory': check_memory()
    }

    success = generate_report(results)

    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
