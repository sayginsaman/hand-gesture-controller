"""
Hand Gesture Media Controller

Required dependencies (install via pip):
pip install opencv-python mediapipe pyautogui

This application uses computer vision to detect hand gestures and control media playback.
When an open palm gesture is detected consistently, it simulates a spacebar keypress.
"""

import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np
from typing import Optional, Tuple, List

# Disable PyAutoGUI failsafe to prevent accidental interruption
pyautogui.FAILSAFE = False

class HandGestureController:
    """
    A class to handle hand gesture detection and media control.
    """
    
    def __init__(self):
        """Initialize the hand gesture controller with MediaPipe and OpenCV."""
        # Initialize MediaPipe hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Detect up to two hands for fullscreen gesture
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Gesture detection parameters
        self.gesture_start_time = None
        self.gesture_threshold = 0.3  # Reduced threshold for quicker response
        self.current_gesture = None
        
        # Media control (open palm) parameters
        self.last_palm_action_time = 0
        self.palm_cooldown = 2.0  # Cooldown for media controls
        self.palm_action_triggered = False
        
        # Fullscreen control (both palms) parameters
        self.last_fullscreen_action_time = 0
        self.fullscreen_cooldown = 3.0  # Longer cooldown for fullscreen toggle
        self.fullscreen_action_triggered = False
        
        # New gesture cooldown parameters
        self.last_gesture_action_time = {}
        self.gesture_cooldowns = {
            'fist': 2.0,           # Mute/unmute
            'two_fingers_up': 1.5,  # Next track
            'two_fingers_down': 1.5, # Previous track
            'thumbs_up': 0.5,      # Volume up (shorter for continuous)
            'thumbs_down': 0.5,    # Volume down (shorter for continuous)
            'swipe_right': 2.0,    # Next tab
            'swipe_left': 2.0      # Previous tab
        }
        self.gesture_action_triggered = {}
        
        # Continuous scrolling parameters - Adjust these for different scrolling behavior
        self.is_scrolling = False
        self.scroll_start_time = None
        self.last_scroll_time = 0
        self.scroll_interval = 0.08  # Time between scroll events (80ms = smooth scrolling)
        self.base_scroll_amount = 2  # Base scroll units per event (lower = slower)
        self.max_scroll_amount = 5  # Maximum scroll units with momentum (higher = faster max speed)
        self.momentum_buildup_time = 2.0  # Time to reach max scroll speed (lower = faster acceleration)
        
        # Volume control parameters
        self.is_volume_changing = False
        self.volume_start_time = None
        self.last_volume_time = 0
        self.volume_interval = 0.2  # Time between volume changes (200ms)
        
        # Swipe detection parameters
        self.hand_positions_history = []
        self.swipe_detection_frames = 10  # Number of frames to track for swipe
        self.swipe_threshold = 0.15  # Minimum distance for swipe detection
        self.swipe_detected = False
        self.last_swipe_time = 0
        
        # Gesture stability tracking
        self.gesture_history = []
        self.gesture_history_size = 3  # Reduced from 5 for faster response
        self.stability_threshold = 0.5  # Reduced from 0.6 for easier detection
        
        # Debug mode (set to True to see detailed gesture detection info)
        self.debug_mode = False
        
        # Initialize camera
        self.cap = None
        
    def setup_camera(self) -> bool:
        """
        Initialize the webcam capture.
        
        Returns:
            bool: True if camera setup successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("Error: Could not open camera")
                print("\nTroubleshooting:")
                print("1. Make sure your webcam is connected and not being used by another application")
                print("2. On macOS, grant camera permissions:")
                print("   - Go to System Preferences > Security & Privacy > Privacy > Camera")
                print("   - Add Terminal (or your Python IDE) to the allowed applications")
                print("3. Try running the script again after granting permissions")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            print("Camera initialized successfully")
            return True
        except Exception as e:
            print(f"Error setting up camera: {e}")
            print("\nTroubleshooting:")
            print("1. Make sure your webcam is connected")
            print("2. Check camera permissions in System Preferences (macOS)")
            print("3. Ensure no other application is using the camera")
            return False
    
    def get_finger_states(self, landmarks) -> List[int]:
        """
        Analyze finger positions to determine which fingers are extended.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            List[int]: List of 5 integers (0 or 1) representing finger states
                      [thumb, index, middle, ring, pinky] - 1 means extended, 0 means folded
        """
        try:
            # Get landmark positions
            landmark_positions = []
            for landmark in landmarks.landmark:
                landmark_positions.append([landmark.x, landmark.y])
            
            fingers_up = []
            
            # Thumb detection (horizontal movement)
            thumb_tip = landmark_positions[4]
            thumb_ip = landmark_positions[3]
            thumb_mcp = landmark_positions[2]
            
            # Determine hand orientation by checking wrist to middle finger direction
            wrist = landmark_positions[0]
            middle_mcp = landmark_positions[9]
            
            # More robust thumb detection
            # Calculate thumb extension by comparing tip to IP joint
            thumb_extended = False
            if middle_mcp[0] > wrist[0]:  # Right hand
                # For right hand, thumb is extended if tip is to the right of IP joint
                thumb_extended = thumb_tip[0] > thumb_ip[0]
            else:  # Left hand
                # For left hand, thumb is extended if tip is to the left of IP joint
                thumb_extended = thumb_tip[0] < thumb_ip[0]
            
            # Additional check: thumb tip should be reasonably far from palm
            thumb_to_wrist_dist = ((thumb_tip[0] - wrist[0])**2 + (thumb_tip[1] - wrist[1])**2)**0.5
            if thumb_to_wrist_dist < 0.08:  # Too close to palm, likely folded
                thumb_extended = False
                
            fingers_up.append(1 if thumb_extended else 0)
            
            # Other four fingers (vertical movement)
            finger_tips = [8, 12, 16, 20]  # Index, Middle, Ring, Pinky
            finger_pips = [6, 10, 14, 18]  # PIP joints
            finger_mcps = [5, 9, 13, 17]   # MCP joints
            
            for tip_idx, pip_idx, mcp_idx in zip(finger_tips, finger_pips, finger_mcps):
                tip = landmark_positions[tip_idx]
                pip = landmark_positions[pip_idx]
                mcp = landmark_positions[mcp_idx]
                
                # Finger is extended if:
                # 1. Tip is above PIP joint (primary condition)
                # 2. AND tip is above MCP joint (additional verification)
                finger_extended = (tip[1] < pip[1]) and (tip[1] < mcp[1])
                
                # Additional check: ensure reasonable extension
                tip_to_mcp_dist = ((tip[0] - mcp[0])**2 + (tip[1] - mcp[1])**2)**0.5
                if tip_to_mcp_dist < 0.06:  # Too close, likely folded
                    finger_extended = False
                
                fingers_up.append(1 if finger_extended else 0)
            
            return fingers_up
            
        except Exception as e:
            print(f"Error in finger state detection: {e}")
            return [0, 0, 0, 0, 0]
    
    def detect_gesture(self, landmarks) -> str:
        """
        Detect the current hand gesture based on finger positions.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            str: Detected gesture name
        """
        fingers = self.get_finger_states(landmarks)
        
        if self.debug_mode:
            print(f"Finger states: {fingers} (T:{fingers[0]}, I:{fingers[1]}, M:{fingers[2]}, R:{fingers[3]}, P:{fingers[4]})")
        
        try:
            # Get landmark positions for advanced gesture detection
            landmark_positions = []
            for landmark in landmarks.landmark:
                landmark_positions.append([landmark.x, landmark.y])
            
            # Open palm: all fingers extended (most reliable)
            if sum(fingers) == 5:
                if self.debug_mode:
                    print("Detected: open_palm (all 5 fingers up)")
                return "open_palm"
            
            # Fist (closed hand): all fingers folded (most reliable)
            elif sum(fingers) == 0:
                if self.debug_mode:
                    print("Detected: fist (all fingers down)")
                return "fist"
            
            # Two fingers (peace sign): ONLY index and middle fingers extended
            elif fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                # Check finger orientation to distinguish up vs down
                index_tip = landmark_positions[8]
                middle_tip = landmark_positions[12]
                wrist = landmark_positions[0]
                
                # Calculate average finger tip height
                avg_finger_y = (index_tip[1] + middle_tip[1]) / 2
                
                # If fingers are significantly above wrist, it's pointing up
                if avg_finger_y < wrist[1] - 0.1:
                    if self.debug_mode:
                        print("Detected: two_fingers_up (peace sign up)")
                    return "two_fingers_up"
                # If fingers are significantly below wrist, it's pointing down
                elif avg_finger_y > wrist[1] + 0.1:
                    if self.debug_mode:
                        print("Detected: two_fingers_down (peace sign down)")
                    return "two_fingers_down"
                else:
                    if self.debug_mode:
                        print("Detected: peace_sign (neutral/sideways)")
                    return "peace_sign"  # Neutral/sideways position
            
            # Index finger up/down: ONLY index finger extended
            elif fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                index_tip = landmark_positions[8]
                wrist = landmark_positions[0]
                
                # Check if pointing up or down
                if index_tip[1] < wrist[1] - 0.05:  # Above wrist
                    if self.debug_mode:
                        print("Detected: index_up")
                    return "index_up"
                elif index_tip[1] > wrist[1] + 0.05:  # Below wrist
                    if self.debug_mode:
                        print("Detected: index_down")
                    return "index_down"
                else:
                    if self.debug_mode:
                        print("Detected: index_up (neutral)")
                    return "index_up"  # Default to up if neutral
            
            # Thumbs up/down: ONLY thumb extended
            elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                thumb_tip = landmark_positions[4]
                wrist = landmark_positions[0]
                
                # Check thumb orientation - use a more reliable method
                # Compare thumb tip with wrist position
                if thumb_tip[1] < wrist[1] - 0.05:  # Thumb tip above wrist
                    if self.debug_mode:
                        print("Detected: thumbs_up")
                    return "thumbs_up"
                elif thumb_tip[1] > wrist[1] + 0.05:  # Thumb tip below wrist
                    if self.debug_mode:
                        print("Detected: thumbs_down")
                    return "thumbs_down"
                else:
                    if self.debug_mode:
                        print("Detected: thumbs_up (neutral)")
                    return "thumbs_up"  # Default to up if neutral
            
            # If we get here, it's an unrecognized pattern
            if self.debug_mode and sum(fingers) > 0:
                print(f"Unrecognized gesture pattern: {fingers}")
                        
        except Exception as e:
            print(f"Error in gesture detection: {e}")
        
        return "none"
    
    def detect_multi_hand_gesture(self, all_hand_landmarks) -> str:
        """
        Detect gestures that require multiple hands.
        
        Args:
            all_hand_landmarks: List of hand landmarks from MediaPipe
            
        Returns:
            str: Detected multi-hand gesture ('both_palms_open', 'none')
        """
        if len(all_hand_landmarks) >= 2:
            # Check if both hands show open palm
            open_palms = 0
            for hand_landmarks in all_hand_landmarks:
                fingers = self.get_finger_states(hand_landmarks)
                if sum(fingers) == 5:  # All fingers extended = open palm
                    open_palms += 1
            
            if open_palms >= 2:
                return "both_palms_open"
        
        return "none"
    
    def get_stable_gesture(self, detected_gesture: str) -> str:
        """
        Apply gesture stability filtering to reduce noise and false detections.
        
        Args:
            detected_gesture (str): The currently detected gesture
            
        Returns:
            str: The stable gesture after filtering
        """
        # Add current gesture to history
        self.gesture_history.append(detected_gesture)
        
        # Keep only recent history
        if len(self.gesture_history) > self.gesture_history_size:
            self.gesture_history.pop(0)
        
        # If we don't have enough history yet, return current gesture
        if len(self.gesture_history) < 3:
            return detected_gesture
        
        # Count occurrences of each gesture in recent history
        gesture_counts = {}
        for gesture in self.gesture_history:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Find the most common gesture
        most_common_gesture = max(gesture_counts, key=gesture_counts.get)
        most_common_count = gesture_counts[most_common_gesture]
        
        # Return the most common gesture if it meets stability threshold
        if most_common_count >= len(self.gesture_history) * self.stability_threshold:
            return most_common_gesture
        else:
            return "none"  # Not stable enough
    
    def calculate_scroll_amount(self) -> int:
        """
        Calculate the scroll amount based on how long the gesture has been held.
        Implements momentum - scrolling gets faster the longer you hold the gesture.
        
        Returns:
            int: The number of scroll units to apply
        """
        if not self.scroll_start_time:
            return self.base_scroll_amount
        
        # Calculate how long the scroll gesture has been active
        scroll_duration = time.time() - self.scroll_start_time
        
        # Apply momentum: gradually increase scroll speed
        momentum_factor = min(scroll_duration / self.momentum_buildup_time, 1.0)
        scroll_amount = self.base_scroll_amount + (self.max_scroll_amount - self.base_scroll_amount) * momentum_factor
        
        return int(scroll_amount)
    
    def handle_gesture_detection(self, detected_gesture: str, multi_hand_gesture: str = "none", swipe_gesture: str = "none") -> None:
        """
        Handle the logic for gesture detection, timing, and action triggering.
        Supports both continuous actions (scrolling, volume) and single-shot actions (media controls).
        
        Args:
            detected_gesture (str): The currently detected single-hand gesture
            multi_hand_gesture (str): The currently detected multi-hand gesture
            swipe_gesture (str): The currently detected swipe gesture
        """
        current_time = time.time()
        
        # Apply stability filtering to reduce noise
        stable_gesture = self.get_stable_gesture(detected_gesture)
        
        # Prioritize gestures: swipe > multi-hand > single-hand
        if swipe_gesture != "none":
            active_gesture = swipe_gesture
        elif multi_hand_gesture != "none":
            active_gesture = multi_hand_gesture
        else:
            active_gesture = stable_gesture
        
        # Handle gesture state changes
        if active_gesture != self.current_gesture:
            # Gesture changed - reset timing and states
            self.gesture_start_time = current_time
            self.current_gesture = active_gesture
            self.palm_action_triggered = False
            self.fullscreen_action_triggered = False
            
            # Reset gesture specific action flags
            for gesture in self.gesture_action_triggered:
                self.gesture_action_triggered[gesture] = False
            
            # Stop any ongoing continuous actions when gesture changes
            if self.is_scrolling and active_gesture not in ["index_up", "index_down"]:
                self.is_scrolling = False
                self.scroll_start_time = None
                print("Scrolling stopped - gesture changed")
            
            if self.is_volume_changing and active_gesture not in ["thumbs_up", "thumbs_down"]:
                self.is_volume_changing = False
                self.volume_start_time = None
                print("Volume control stopped - gesture changed")
        
        # Handle active gestures
        if active_gesture != "none" and self.gesture_start_time:
            gesture_duration = current_time - self.gesture_start_time
            
            # Handle both palms open gesture (fullscreen toggle)
            if (active_gesture == "both_palms_open" and 
                gesture_duration >= self.gesture_threshold and
                not self.fullscreen_action_triggered and
                current_time - self.last_fullscreen_action_time >= self.fullscreen_cooldown):
                
                self.trigger_fullscreen()
                self.last_fullscreen_action_time = current_time
                self.fullscreen_action_triggered = True
            
            # Handle open palm gesture (play/pause)
            elif (active_gesture == "open_palm" and 
                gesture_duration >= self.gesture_threshold and
                not self.palm_action_triggered and
                current_time - self.last_palm_action_time >= self.palm_cooldown):
                
                self.trigger_spacebar()
                self.last_palm_action_time = current_time
                self.palm_action_triggered = True
            
            # Handle continuous scrolling gestures (index finger up/down)
            elif active_gesture in ["index_up", "index_down"] and gesture_duration >= self.gesture_threshold:
                if not self.is_scrolling:
                    self.is_scrolling = True
                    self.scroll_start_time = current_time
                    self.last_scroll_time = current_time
                    print(f"Started continuous scrolling: {active_gesture}")
                
                if current_time - self.last_scroll_time >= self.scroll_interval:
                    scroll_amount = self.calculate_scroll_amount()
                    
                    if active_gesture == "index_up":
                        self.trigger_continuous_scroll_up(scroll_amount)
                    elif active_gesture == "index_down":
                        self.trigger_continuous_scroll_down(scroll_amount)
                    
                    self.last_scroll_time = current_time
            
            # Handle continuous volume control gestures (thumbs up/down)
            elif active_gesture in ["thumbs_up", "thumbs_down"] and gesture_duration >= self.gesture_threshold:
                if not self.is_volume_changing:
                    self.is_volume_changing = True
                    self.volume_start_time = current_time
                    self.last_volume_time = current_time
                    print(f"Started volume control: {active_gesture}")
                
                if current_time - self.last_volume_time >= self.volume_interval:
                    if active_gesture == "thumbs_up":
                        self.trigger_volume_up()
                    elif active_gesture == "thumbs_down":
                        self.trigger_volume_down()
                    
                    self.last_volume_time = current_time
            
            # Handle single shot gestures with cooldowns
            elif active_gesture in self.gesture_cooldowns:
                gesture_key = active_gesture
                cooldown = self.gesture_cooldowns[gesture_key]
                last_action_time = self.last_gesture_action_time.get(gesture_key, 0)
                action_triggered = self.gesture_action_triggered.get(gesture_key, False)
                
                if (gesture_duration >= self.gesture_threshold and
                    not action_triggered and
                    current_time - last_action_time >= cooldown):
                    
                    # Trigger appropriate action
                    if active_gesture == "fist":
                        self.trigger_mute()
                    elif active_gesture == "two_fingers_up":
                        self.trigger_next_track()
                    elif active_gesture == "two_fingers_down":
                        self.trigger_previous_track()
                    elif active_gesture == "swipe_right":
                        self.trigger_next_tab()
                    elif active_gesture == "swipe_left":
                        self.trigger_previous_tab()
                    
                    self.last_gesture_action_time[gesture_key] = current_time
                    self.gesture_action_triggered[gesture_key] = True
        
        if active_gesture == "none":
            if self.is_scrolling:
                self.is_scrolling = False
                self.scroll_start_time = None
                print("Scrolling stopped - no gesture detected")
            
            if self.is_volume_changing:
                self.is_volume_changing = False
                self.volume_start_time = None
                print("Volume control stopped - no gesture detected")
            
            # Reset all timing
            self.gesture_start_time = None
            self.current_gesture = None
            self.palm_action_triggered = False
            self.fullscreen_action_triggered = False
            
            # Reset gesture-specific action flags
            for gesture in self.gesture_action_triggered:
                self.gesture_action_triggered[gesture] = False
    
    def trigger_spacebar(self) -> None:
        """
        Simulate a spacebar keypress to control media playback.
        """
        try:
            pyautogui.press('space')
            print("Spacebar triggered - Media play/pause")
        except Exception as e:
            print(f"Error triggering spacebar: {e}")
    
    def trigger_fullscreen(self) -> None:
        """
        Simulate a fullscreen toggle keypress.
        Uses 'f' key which is the standard fullscreen toggle for video players like YouTube, VLC, etc.
        This is more reliable than F11 on macOS which can trigger Mission Control.
        """
        try:
            pyautogui.press('f')
            print("Fullscreen toggle triggered (F key)")
        except Exception as e:
            print(f"Error triggering fullscreen: {e}")
    
    def trigger_mute(self) -> None:
        """
        Simulate mute/unmute toggle.
        Uses the standard mute key shortcut.
        """
        try:
            pyautogui.press('m')
            print("Mute/unmute triggered")
        except Exception as e:
            print(f"Error triggering mute: {e}")
    
    def trigger_next_track(self) -> None:
        """
        Simulate next track/video command.
        Uses media next key or right arrow for video players.
        """
        try:
            pyautogui.press('shift', 'n')
            print("Next track triggered")
        except Exception as e:
            try:
                pyautogui.press('shift', 'n')
                print("Next track triggered (shift + n)")
            except Exception as e2:
                print(f"Error triggering next track: {e2}")
    
    def trigger_previous_track(self) -> None:
        """
        Simulate previous track/video command.
        Uses media previous key or left arrow for video players.
        """
        try:
            # Try media key first, fallback to left arrow
            pyautogui.press('prevtrack')
            print("Previous track triggered")
        except Exception as e:
            try:
                pyautogui.press('left')
                print("Previous track triggered (left arrow)")
            except Exception as e2:
                print(f"Error triggering previous track: {e2}")
    
    def trigger_volume_up(self) -> None:
        """
        Simulate volume up command.
        """
        try:
            pyautogui.press('up')
            print("Volume up triggered")
        except Exception as e:
            print(f"Error triggering volume up: {e}")
    
    def trigger_volume_down(self) -> None:
        """
        Simulate volume down command.
        """
        try:
            pyautogui.press('down')
            print("Volume down triggered")
        except Exception as e:
            print(f"Error triggering volume down: {e}")
    

    def trigger_next_tab(self) -> None:
        """
        Simulate next tab command.
        Uses Cmd+Shift+] (macOS) or Ctrl+Tab.
        """
        try:
            # macOS shortcut
            pyautogui.hotkey('command', 'shift', ']')
            print("Next tab triggered (Cmd+Shift+])")
        except Exception as e:
            try:
                # Alternative shortcut
                pyautogui.hotkey('ctrl', 'tab')
                print("Next tab triggered (Ctrl+Tab)")
            except Exception as e2:
                print(f"Error triggering next tab: {e2}")
    
    def trigger_previous_tab(self) -> None:
        """
        Simulate previous tab command.
        Uses Cmd+Shift+[ (macOS) or Ctrl+Shift+Tab.
        """
        try:
            # macOS shortcut
            pyautogui.hotkey('command', 'shift', '[')
            print("Previous tab triggered (Cmd+Shift+[)")
        except Exception as e:
            try:
                # Alternative shortcut
                pyautogui.hotkey('ctrl', 'shift', 'tab')
                print("Previous tab triggered (Ctrl+Shift+Tab)")
            except Exception as e2:
                print(f"Error triggering previous tab: {e2}")
    
    def trigger_continuous_scroll_down(self, scroll_amount: int) -> None:
        """
        Simulate continuous scrolling down on the page/document.
        
        Args:
            scroll_amount (int): Number of scroll units to apply
        """
        try:
            # Negative values scroll down in PyAutoGUI
            pyautogui.scroll(-scroll_amount)
            # Only print occasionally to avoid spam
            if time.time() - getattr(self, '_last_down_print', 0) > 1.0:
                print(f"Continuous scroll down (speed: {scroll_amount})")
                self._last_down_print = time.time()
        except Exception as e:
            print(f"Error in continuous scroll down: {e}")
    
    def trigger_continuous_scroll_up(self, scroll_amount: int) -> None:
        """
        Simulate continuous scrolling up on the page/document.
        
        Args:
            scroll_amount (int): Number of scroll units to apply
        """
        try:
            # Positive values scroll up in PyAutoGUI
            pyautogui.scroll(scroll_amount)
            # Only print occasionally to avoid spam
            if time.time() - getattr(self, '_last_up_print', 0) > 1.0:
                print(f"Continuous scroll up (speed: {scroll_amount})")
                self._last_up_print = time.time()
        except Exception as e:
            print(f"Error in continuous scroll up: {e}")
    
    # Legacy methods for backward compatibility (not used in continuous mode)
    def trigger_scroll_down(self) -> None:
        """Legacy method - kept for compatibility."""
        self.trigger_continuous_scroll_down(self.base_scroll_amount)
    
    def trigger_scroll_up(self) -> None:
        """Legacy method - kept for compatibility."""
        self.trigger_continuous_scroll_up(self.base_scroll_amount)
    
    def draw_info_overlay(self, frame: np.ndarray, detected_gesture: str) -> np.ndarray:
        """
        Draw information overlay on the video frame.
        
        Args:
            frame: The video frame to draw on
            detected_gesture: The currently detected gesture
            
        Returns:
            np.ndarray: Frame with overlay information
        """
        # Define gesture display information
        gesture_info = {
            "open_palm": ("Open Palm - Play/Pause", (0, 255, 0)),
            "fist": ("Fist - Mute/Unmute", (255, 0, 0)),
            "index_up": ("Index Up - Scroll Up", (255, 165, 0)),
            "index_down": ("Index Down - Scroll Down", (255, 100, 0)),
            "thumbs_up": ("Thumbs Up - Volume Up", (0, 255, 255)),
            "thumbs_down": ("Thumbs Down - Volume Down", (0, 200, 255)),
            "two_fingers_up": ("Peace Sign ✌️ - Next Track", (255, 255, 0)),
            "two_fingers_down": ("Two Down - Previous Track", (255, 200, 0)),
            "swipe_right": ("Swipe Right - Next Tab", (255, 128, 255)),
            "swipe_left": ("Swipe Left - Previous Tab", (200, 128, 255)),
            "both_palms_open": ("Both Palms - Fullscreen", (255, 0, 255)),
            "none": ("No Gesture", (0, 0, 255))
        }
        
        # Draw status information
        status_text, status_color = gesture_info.get(detected_gesture, ("Unknown", (128, 128, 128)))
        cv2.putText(frame, f"Gesture: {status_text}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Show gesture timing progress
        if self.gesture_start_time is not None and detected_gesture != "none":
            elapsed = time.time() - self.gesture_start_time
            progress = min(elapsed / self.gesture_threshold, 1.0)
            cv2.putText(frame, f"Hold Progress: {progress:.1%}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Show continuous action status
        y_offset = 90
        if self.is_scrolling:
            scroll_speed = self.calculate_scroll_amount()
            scroll_duration = time.time() - self.scroll_start_time if self.scroll_start_time else 0
            cv2.putText(frame, f"Scrolling: Speed {scroll_speed} | Duration {scroll_duration:.1f}s", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            y_offset += 25
        
        if self.is_volume_changing:
            volume_duration = time.time() - self.volume_start_time if self.volume_start_time else 0
            cv2.putText(frame, f"Volume Control: Duration {volume_duration:.1f}s", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            y_offset += 25
        
        # Show finger states for debugging (optional)
        if hasattr(self, '_last_finger_states'):
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            finger_text = " | ".join([f"{name}: {'↑' if state else '↓'}" 
                                    for name, state in zip(finger_names, self._last_finger_states)])
            cv2.putText(frame, finger_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Show compact instructions
        instructions = [
            "🖐 Open Palm = Play/Pause | ✊ Fist = Mute | ✌️ Peace = Next/Prev Track",
            "☝️ Index Up/Down = Scroll | 👍👎 Thumbs = Volume",
            "↔️ Swipe Left/Right = Previous/Next Tab | 🙌 Both Palms = Fullscreen",
            "Press 'q' to quit"
        ]
        
        start_y = frame.shape[0] - 60
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, start_y + i * 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return frame
    
    def run(self) -> None:
        """
        Main application loop for hand gesture detection and control.
        """
        if not self.setup_camera():
            return
        
        print("Hand Gesture Controller started")
        print("Available gestures:")
        print("📱 MEDIA CONTROLS:")
        print("  - 🖐 Open Palm (0.3s) = Play/Pause")
        print("  - ✊ Fist (0.3s) = Mute/Unmute")
        print("  - ✌️ Peace Sign Up (0.3s) = Next Track")
        print("  - ✌️ Peace Sign Down (0.3s) = Previous Track")
        print("📜 SCROLLING:")
        print("  - ☝️ Index Up (hold) = Continuous Scroll Up")
        print("  - ☝️ Index Down (hold) = Continuous Scroll Down")
        print("🔊 VOLUME:")
        print("  - 👍 Thumbs Up (hold) = Volume Up")
        print("  - 👎 Thumbs Down (hold) = Volume Down")
        print("🖥️ NAVIGATION:")
        print("  - ↔️ Swipe Left = Previous Tab")
        print("  - ↔️ Swipe Right = Next Tab")
        print("  - 🙌 Both Palms (0.3s) = Fullscreen Toggle")
        print("\nPress 'q' to quit")
        
        try:
            while True:
                # Read frame from camera
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame for hand detection
                results = self.hands.process(rgb_frame)
                
                # Initialize gesture detection
                detected_gesture = "none"
                multi_hand_gesture = "none"
                swipe_gesture = "none"
                
                # Process hand landmarks if detected
                if results.multi_hand_landmarks:
                    # Draw all hand landmarks on the frame
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Detect multi-hand gestures first (higher priority)
                    multi_hand_gesture = self.detect_multi_hand_gesture(results.multi_hand_landmarks)
                    
                    # If no multi-hand gesture, detect single-hand gesture from first hand
                    if multi_hand_gesture == "none" and len(results.multi_hand_landmarks) > 0:
                        first_hand = results.multi_hand_landmarks[0]
                        detected_gesture = self.detect_gesture(first_hand)
                        
                        # Detect swipe gestures (only if palm is sideways/flat)
                        if detected_gesture in ["none", "open_palm"]:
                            swipe_gesture = self.detect_swipe(first_hand)
                        
                        # Store finger states for debugging display (optional)
                        self._last_finger_states = self.get_finger_states(first_hand)
                else:
                    # Clear hand position history when no hands detected
                    self.hand_positions_history.clear()
                
                # Handle gesture detection logic and trigger actions
                self.handle_gesture_detection(detected_gesture, multi_hand_gesture, swipe_gesture)
                
                # Determine which gesture to display (prioritize swipe > multi-hand > single-hand)
                if swipe_gesture != "none":
                    display_gesture = swipe_gesture
                elif multi_hand_gesture != "none":
                    display_gesture = multi_hand_gesture
                else:
                    display_gesture = detected_gesture
                
                # Draw information overlay with current gesture status
                frame = self.draw_info_overlay(frame, display_gesture)
                
                # Display the frame
                cv2.imshow('Hand Gesture Controller', frame)
                
                # Check for quit command
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nApplication interrupted by user")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """
        Clean up resources when the application exits.
        """
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed successfully")

    def detect_swipe(self, landmarks) -> str:
        """
        Detect horizontal swipe gestures based on hand movement over time.
        
        Args:
            landmarks: MediaPipe hand landmarks
            
        Returns:
            str: Detected swipe gesture ('swipe_left', 'swipe_right', 'none')
        """
        try:
            # Get center of palm (wrist position)
            wrist = landmarks.landmark[0]
            current_position = [wrist.x, wrist.y]
            
            # Add current position to history
            self.hand_positions_history.append(current_position)
            
            # Keep only recent positions
            if len(self.hand_positions_history) > self.swipe_detection_frames:
                self.hand_positions_history.pop(0)
            
            # Need enough history to detect swipe
            if len(self.hand_positions_history) < self.swipe_detection_frames:
                return "none"
            
            # Calculate horizontal movement
            start_x = self.hand_positions_history[0][0]
            end_x = self.hand_positions_history[-1][0]
            horizontal_movement = end_x - start_x
            
            # Check if movement exceeds threshold
            if abs(horizontal_movement) > self.swipe_threshold:
                current_time = time.time()
                
                # Prevent rapid swipe detection
                if current_time - self.last_swipe_time > 1.0:
                    self.last_swipe_time = current_time
                    
                    if horizontal_movement > 0:
                        return "swipe_right"
                    else:
                        return "swipe_left"
            
            return "none"
            
        except Exception as e:
            print(f"Error in swipe detection: {e}")
            return "none"

def main():
    """
    Main function to run the hand gesture controller application.
    """
    try:
        controller = HandGestureController()
        controller.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Make sure you have installed all required dependencies:")
        print("pip install opencv-python mediapipe pyautogui")

if __name__ == "__main__":
    main() 