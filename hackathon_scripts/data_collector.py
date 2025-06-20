import sys
import os
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# Add project root to sys.path to allow importing lerobot modules
sys.path.append(str(Path(__file__).parent.parent))

from lerobot.common.robots.realsense.config_realsense import RealSenseClientConfig
from lerobot.common.robots.realsense.realsense_client import RealSenseClient

# --- Configuration ---
DATA_DIR = "data"

def main():
    """
    A script to collect hand gesture data using a RealSense camera and MediaPipe.
    """
    # --- Setup ---
    gesture_name = input("Enter the name for the gesture you are recording (e.g., 'call_mom'): ").strip().lower().replace(" ", "_")
    if not gesture_name:
        print("Gesture name cannot be empty. Exiting.")
        return

    gesture_dir = Path(DATA_DIR) / gesture_name
    gesture_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data will be saved in: {gesture_dir}")

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize RealSense Client
    print("Initializing RealSense client...")
    robot = RealSenseClient(RealSenseClientConfig())
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to the camera. Exiting.")
        return

    # --- Main Loop ---
    print("\nReady to record. Press 'r' to START/STOP recording.")
    print("Press 'q' to QUIT.")

    is_recording = False
    recorded_data = []

    try:
        while True:
            obs = robot.get_observation()
            frame_tensor = obs.get("observation.image")
            if frame_tensor is None:
                continue
            
            frame = frame_tensor.numpy()
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = hands.process(image_rgb)
            image_rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                if is_recording:
                    frame_landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    recorded_data.append(frame_landmarks)

            if is_recording:
                cv2.putText(frame, "RECORDING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            
            if key == ord('r'):
                if not is_recording:
                    print("Started recording...")
                    is_recording = True
                    recorded_data = []
                else:
                    print("Stopped recording.")
                    is_recording = False
                    if recorded_data:
                        timestamp = int(time.time())
                        filepath = gesture_dir / f"{timestamp}.json"
                        with open(filepath, 'w') as f:
                            json.dump(recorded_data, f, indent=2)
                        print(f"Saved {len(recorded_data)} frames to {filepath}")
                        recorded_data = []

    finally:
        print("Disconnecting...")
        hands.close()
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
