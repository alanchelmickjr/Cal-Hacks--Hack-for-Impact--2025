import sys
from pathlib import Path

import cv2

# This allows us to import the new realsense module without reinstalling lerobot
sys.path.append(str(Path(__file__).parent.parent))

from lerobot.common.robots.realsense.config_realsense import RealSenseClientConfig
from lerobot.common.robots.realsense.realsense_client import RealSenseClient


def main():
    """
    Initializes the RealSense client, connects to the camera,
    and displays the video feed in a window.
    """
    print("Initializing RealSense client...")
    config = RealSenseClientConfig()
    robot = RealSenseClient(config)

    print("Connecting to camera...")
    robot.connect()

    if not robot.is_connected:
        print("Failed to connect to the camera. Exiting.")
        return

    print("Streaming video... Press 'q' to quit.")
    try:
        while True:
            obs = robot.get_observation()
            image_tensor = obs.get("observation.image")

            if image_tensor is not None:
                # Convert torch tensor to numpy array for display
                image_np = image_tensor.numpy()
                cv2.imshow("RealSense Feed", image_np)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("Disconnecting camera...")
        robot.disconnect()
        cv2.destroyAllWindows()
        print("Stream stopped.")


if __name__ == "__main__":
    main()
