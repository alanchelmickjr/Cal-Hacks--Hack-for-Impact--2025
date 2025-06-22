"""Tele-operation demo that streams a camera feed on the follower side.

Hardware:
    • Koch (SO101) red follower arm with an attached OpenCV-compatible camera.
    • Koch (SO101) blue leader arm for teleoperation.

Environment variables (override defaults):
    LEADER_PORT       serial port of the blue leader arm
    FOLLOWER_PORT     serial port of the red follower arm
    CAMERA_INDEX      numeric camera index or path recognised by OpenCV (default: 0)

Run:
    python hackathon_scripts/teleop_camera_test.py
"""

from __future__ import annotations

import os
import signal
import sys
from threading import Thread
from typing import Any

from lerobot.common.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.common.teleoperators.koch_leader import KochLeaderConfig, KochLeader
from lerobot.common.robots.koch_follower import KochFollowerConfig, KochFollower

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def build_camera_config() -> dict[str, OpenCVCameraConfig]:
    """Return a dict understood by KochFollower for camera attachment."""
    idx_or_path: Any = os.getenv("CAMERA_INDEX", "0")
    # Cast to int if it looks like a digit, else keep as string path
    if str(idx_or_path).isdigit():
        idx_or_path = int(idx_or_path)
    return {
        "front": OpenCVCameraConfig(index_or_path=idx_or_path, width=1920, height=1080, fps=30)
    }


def build_robot() -> KochFollower:
    robot_config = KochFollowerConfig(
        port=os.getenv("FOLLOWER_PORT", "/dev/tty.usbmodem585A0076841"),
        id="my_red_robot_arm",
        cameras=build_camera_config(),
    )
    return KochFollower(robot_config)


def build_teleop() -> KochLeader:
    teleop_config = KochLeaderConfig(
        port=os.getenv("LEADER_PORT", "/dev/tty.usbmodem58760431551"),
        id="my_blue_leader_arm",
    )
    return KochLeader(teleop_config)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def teleop_loop(robot: KochFollower, teleop: KochLeader) -> None:
    try:
        while True:
            observation = robot.get_observation()  # noqa: F841  # keep for future use
            action = teleop.get_action()
            robot.send_action(action)
    except KeyboardInterrupt:
        pass  # handled in outer scope


def main() -> None:
    robot = build_robot()
    teleop = build_teleop()

    robot.connect()
    teleop.connect()

    # Clean shutdown on Ctrl-C
    def _graceful_exit(signum: int, frame):  # noqa: D401, N802, unused-argument
        print("\nStopping teleoperation…")
        teleop.disconnect()
        robot.disconnect()
        sys.exit(0)

    signal.signal(signal.SIGINT, _graceful_exit)

    print("Tele-operation with camera started. Press Ctrl-C to quit.")
    teleop_loop(robot, teleop)


if __name__ == "__main__":
    main()
