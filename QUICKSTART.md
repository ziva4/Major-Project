# Quick Start Guide - Violence Detection System

## 🚀 Get Running in 5 Minutes

### For Windows Users:
1. Navigate to the `backend` folder
2. Double-click `run.bat`
3. Wait for model training (first run only)
4. Open http://localhost:5000 in your browser

### For Linux/macOS Users:
```bash
cd backend
./run.sh
```

## 📊 What You'll See

1. **Live Camera Feed** - Real-time video from your PC camera
2. **Confidence Bar** - Shows violence detection level (0-100%)
3. **Status Indicator** - Green (Safe) or Red (Danger)
4. **Alerts List** - All detected incidents with timestamps
5. **Settings Panel** - Adjust sensitivity with slider

## 🎮 Try It Out

### To Test Normal Activity:
- Wave at camera normally
- Move around calmly
- Should stay green with low confidence

### To Simulate High Confidence:
- Make rapid, sudden movements
- Wave hands quickly and aggressively
- Camera feed should turn red if confidence > threshold

### Alert Acknowledgment:
- When alert appears, click "ACKNOWLEDGE"
- Status changes from "ACTIVE" to "ACKNOWLEDGED"

## ⚙️ Configuration

### Adjust Sensitivity:
- **Dashboard**: Use confidence slider (0-100%)
- **Default**: 60% (balanced between detection and false positives)
- **High Security**: 40-50% (more alerts)
- **High Tolerance**: 70-80% (fewer alerts)

### Change Port (if 5000 is busy):
Edit `backend/app.py`, line ~105:
```python
socketio.run(app, host='0.0.0.0', port=5001)  # Change 5000 to your port
```

## 📁 Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Flask server |
| `video_processor.py` | Camera & detection processing |
| `train_model.py` | ML model training |
| `templates/index.html` | Dashboard interface |

## 🔧 Troubleshooting

### Camera Not Working?
- Check if camera is plugged in
- Check Windows/macOS privacy settings
- Try index 1 in `video_processor.py` line 20: `self.cap = cv2.VideoCapture(1)`

### Port Already in Use?
- Change port number in `app.py`
- Or: `lsof -i :5000` (macOS) / `netstat -ano | findstr :5000` (Windows)

### Model Training Stuck?
- Check internet connection (downloading TensorFlow)
- Increase available RAM
- Try reducing samples: change `samples=500` in `train_model.py`

### High CPU Usage?
- Reduce frame resolution in `video_processor.py` line 23
- Lower FPS with sleep increase in line 62

## 📈 Performance Tips

For Better Speed:
```
✓ Close other applications
✓ Increase threshold (fewer detections)
✓ Use good lighting for camera
✓ Use dedicated GPU (if available)
```

For Better Accuracy:
```
✓ Lower threshold (more detections)
✓ Retrain with real violence/normal data
✓ Improve camera angle and focus
✓ Ensure 60+ FPS camera feed
```

## 📝 Understanding the Dashboard

### Status Indicator Colors:
- 🟢 **Green** = Safe (Normal activity)
- 🔴 **Red** = Warning (Violence detected)

### Confidence Bar:
- **0-30%** = Definitely safe
- **30-60%** = Suspicious activity
- **60-100%** = High confidence violence

### Alerts Table Columns:
- **Time** = When violence was detected
- **Confidence** = Detection certainty (%)
- **Status** = ACTIVE or ACKNOWLEDGED
- **Acknowledge** = Button to mark as reviewed

## 🎯 Next Steps

1. **First Run**: Just observe normal activity
2. **Test Sensitivity**: Adjust slider to find optimal threshold
3. **Train Your Model**: Replace synthetic data with real videos
4. **Deploy**: Run on dedicated surveillance PC
5. **Police Integration**: Connect to actual police systems (advanced)

## 📞 Support Resources

- **Error in Console?** Read the error message carefully
- **Model Not Training?** Check internet + RAM
- **Camera Issues?** Grant camera permissions in Settings
- **Still Stuck?** Contact technical support with:
  - Error message
  - Your OS and Python version
  - What you were doing when it broke

## 📊 System Requirements Checklist

- [ ] Python 3.8+ installed
- [ ] Webcam connected and working
- [ ] Internet connection (for first setup)
- [ ] 4GB+ RAM available
- [ ] 2GB free disk space
- [ ] Port 5000 available

## 🎓 Learning Resources

### Understanding the Model:
1. Features extracted from each frame (color, texture, motion)
2. Neural network classifies: Normal (0) or Violent (1)
3. Confidence score from 0-1 is converted to percentage
4. Smoothing across 5 frames reduces false positives

### Custom Training:
1. Collect 500+ videos of violence
2. Collect 500+ videos of normal activity
3. Extract frames and label them
4. Modify `train_model.py` to use real data
5. Run training

## 🔒 Security Notes

⚠️ **Before Using in Real Surveillance:**
- Enable HTTPS encryption
- Add authentication/login
- Encrypt stored alert data
- Log all access and actions
- Comply with local privacy laws
- Get proper authorization

## 🌟 Pro Tips

1. **Multi-Camera Setup**: Run multiple instances on different ports
2. **Always On**: Use `screen` or `tmux` on Linux to keep running
3. **Remote Access**: Set `host='0.0.0.0'` and access from other devices
4. **Backup Alerts**: Periodically export alert data
5. **Test Regularly**: Verify system is working with test runs

---

**Ready to go?** Run `run.bat` (Windows) or `./run.sh` (Linux/macOS) and open http://localhost:5000

Good luck! 🚨
