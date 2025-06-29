#!/usr/bin/env python3
"""
Test script for hand gesture detection with debug output enabled.
This will help identify accuracy issues with gesture detection.
"""

from hand_gesture_controller import HandGestureController

def main():
    """Run the gesture controller with debug mode enabled."""
    print("Starting Hand Gesture Controller in DEBUG MODE")
    print("This will show detailed gesture detection information.")
    print("Use this to understand what gestures are being detected and why.")
    print("-" * 60)
    
    try:
        controller = HandGestureController()
        # Enable debug mode
        controller.debug_mode = True
        controller.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install opencv-python mediapipe pyautogui")

if __name__ == "__main__":
    main() 