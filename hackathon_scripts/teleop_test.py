"""Teleoperation script to mirror actions from a SO101 leader arm to a SO101 follower arm.

Usage:
    python teleop_test.py

The script first looks for two environment variables:
    - `LEADER_PORT`: serial device for the blue leader arm
    - `FOLLOWER_PORT`: serial device for the red follower arm

If they are not set, it falls back to default `usbmodem` values that may or may
not match your system.
"""

import os

from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import (
    SO101FollowerConfig,
    SO101Follower,
)


def main() -> None:
    """Spin up teleoperation where the leader arm drives the follower arm."""
    # Configure follower (red robot arm)
    robot_config = SO101FollowerConfig(
        port=os.getenv("FOLLOWER_PORT", "/dev/tty.usbmodem58760431541"),
        id="my_red_robot_arm",
    )

    # Configure leader (blue teleop device)
    teleop_config = SO101LeaderConfig(
        port=os.getenv("LEADER_PORT", "/dev/tty.usbmodem58760431551"),
        id="my_blue_leader_arm",
    )

    # Instantiate devices
    robot = SO101Follower(robot_config)
    teleop_device = SO101Leader(teleop_config)

    # Establish connections
    robot.connect()
    teleop_device.connect()

    print("Teleoperation started — move the blue arm to control the red arm.")
    try:
        while True:
            # Read leader action and forward it to the follower.
            action = teleop_device.get_action()
            robot.send_action(action)
    except KeyboardInterrupt:
        print("\nTeleoperation stopped by user.")
    finally:
        # Gracefully disconnect on exit.
        teleop_device.disconnect()
        robot.disconnect()


if __name__ == "__main__":
    main()
