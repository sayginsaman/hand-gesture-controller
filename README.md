# Hand Gesture Desktop Controller

A comprehensive Python application that uses computer vision to detect hand gestures and control various desktop functions including media playback, volume control, scrolling, and navigation.

## Features

### 📱 Media Controls
- **🖐 Open Palm** (0.3s hold) → Play/Pause media (spacebar)
- **✊ Fist** (0.3s hold) → Mute/Unmute audio
- **✌️ Peace Sign Up** (0.3s hold) → Next track/video
- **✌️ Peace Sign Down** (0.3s hold) → Previous track/video

### 📜 Scrolling
- **☝️ Index Finger Up** (hold continuously) → Smooth scroll up with momentum
- **☝️ Index Finger Down** (hold continuously) → Smooth scroll down with momentum

### 🔊 Volume Control
- **👍 Thumbs Up** (hold continuously) → Increase volume
- **👎 Thumbs Down** (hold continuously) → Decrease volume

### 🖥️ Navigation & Display
- **↔️ Swipe Left** → Previous tab/slide
- **↔️ Swipe Right** → Next tab/slide  
- **🙌 Both Palms Open** (0.3s hold) → Toggle fullscreen

## Technical Features

- **Real-time hand detection** using MediaPipe
- **Gesture stability filtering** to reduce false detections
- **Momentum-based scrolling** (speed increases with hold duration)
- **Continuous volume control** with smooth adjustment
- **Multi-hand gesture support** for advanced controls
- **Swipe detection** for horizontal navigation
- **Visual feedback overlay** showing gesture status and timing
- **Comprehensive error handling** and camera permission guidance

## Requirements

- Python 3.8 to 3.12 (MediaPipe compatibility)
- Webcam/camera access
- macOS, Windows, or Linux

## Installation

### Option 1: Automated Setup (Recommended)
```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python hand_gesture_controller.py
```

## Dependencies

- **opencv-python**: Computer vision and camera access
- **mediapipe**: Hand detection and landmark tracking
- **pyautogui**: Desktop automation and input simulation
- **numpy**: Numerical computations

## Usage

1. **Start the application**: Run `python hand_gesture_controller.py`
2. **Grant camera permissions** when prompted (especially on macOS)
3. **Position yourself** in front of the camera with good lighting
4. **Perform gestures** as shown in the on-screen guide
5. **Press 'q'** to quit the application

### Gesture Tips

- **Hold gestures steady** for 0.3 seconds to trigger single-shot actions
- **Keep holding** scrolling and volume gestures for continuous control
- **Use clear, distinct gestures** - avoid partial or unclear hand positions
- **Ensure good lighting** for better hand detection
- **Stay within camera frame** with hands visible

## Keyboard Shortcuts Used

| Gesture | Primary Shortcut | Fallback | Platform |
|---------|------------------|----------|----------|
| Play/Pause | `Space` | - | All |
| Mute | `M ` | - | All |
| Next Track | `Shift + N` | - | All |
| Previous Track | `Shift + P` | `Left Arrow` | All |
| Volume Up/Down | `Up Arrow Key/Down Arrow Key` | - | All |

| Next Tab | `Cmd+Shift+]` | `Ctrl+Tab` | macOS/Windows |
| Previous Tab | `Cmd+Shift+[` | `Ctrl+Shift+Tab` | macOS/Windows |
| Fullscreen | `F` | - | All |

## Troubleshooting

### Camera Issues
- **macOS**: Grant camera permissions in System Preferences > Security & Privacy > Privacy > Camera
- **Windows**: Check camera permissions in Settings > Privacy > Camera
- **Linux**: Ensure user is in `video` group: `sudo usermod -a -G video $USER`

### Python Version Issues
- MediaPipe requires Python 3.8-3.12
- Use `python3 --version` to check your version
- Consider using `pyenv` or virtual environments for version management

### Performance Issues
- Ensure good lighting for better hand detection
- Close other applications using the camera
- Lower camera resolution if needed (modify `setup_camera()` method)

### Gesture Recognition Issues
- Make clear, distinct gestures
- Hold gestures steady for the required duration
- Ensure hands are fully visible in camera frame
- Check the finger state display for debugging

## Customization

### Adjust Gesture Sensitivity
Edit these parameters in `hand_gesture_controller.py`:
```python
self.gesture_threshold = 0.3  # Time to hold gesture (seconds)
self.stability_threshold = 0.6  # Gesture stability filtering
```

### Modify Scrolling Behavior
```python
self.scroll_interval = 0.08  # Time between scroll events
self.base_scroll_amount = 2  # Base scroll speed
self.max_scroll_amount = 5  # Maximum scroll speed with momentum
```

### Change Volume Control Speed
```python
self.volume_interval = 0.2  # Time between volume changes
```

## Architecture

The application uses a class-based architecture with the following key components:

- **`HandGestureController`**: Main controller class
- **Gesture Detection**: MediaPipe-based hand landmark analysis
- **Stability Filtering**: Reduces noise and false detections
- **Action Handlers**: Separate methods for each gesture action
- **Visual Feedback**: Real-time overlay showing gesture status
- **Error Handling**: Comprehensive error management and recovery

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Acknowledgments

- **MediaPipe** by Google for hand detection
- **OpenCV** for computer vision capabilities
- **PyAutoGUI** for desktop automation 
