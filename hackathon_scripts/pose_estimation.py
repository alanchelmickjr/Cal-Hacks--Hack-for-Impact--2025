import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# This allows us to import the new realsense module without reinstalling lerobot
sys.path.append(str(Path(__file__).parent.parent))

from lerobot.common.robots.realsense.config_realsense import RealSenseClientConfig
from lerobot.common.robots.realsense.realsense_client import RealSenseClient


def main():
    """
    Initializes the RealSense client, runs MediaPipe hand tracking on the video feed,
    and displays the results in a window.
    """
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    print("Initializing RealSense client...")
    config = RealSenseClientConfig()
    robot = RealSenseClient(config)

    print("Connecting to camera...")
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to the camera. Exiting.")
        return

    print("Streaming video with pose estimation... Press 'q' to quit.")
    try:
        while True:
            obs = robot.get_observation()
            image_tensor = obs.get("observation.image")

            if image_tensor is not None:
                # Convert torch tensor to numpy array
                frame = image_tensor.numpy()

                # MediaPipe expects RGB, but OpenCV provides BGR. RealSense gives BGR.
                # So we need to convert BGR to RGB for MediaPipe.
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)

                # Draw the hand annotations on the original frame.
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cv2.imshow("Pose Estimation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Disconnecting camera...")
        hands.close()
        robot.disconnect()
        cv2.destroyAllWindows()
        print("Stream stopped.")


if __name__ == "__main__":
    main()
