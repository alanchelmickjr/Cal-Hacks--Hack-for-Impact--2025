# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import cached_property
from typing import Any

import numpy as np
import pyrealsense2 as rs
import torch

from lerobot.common.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError
from lerobot.common.robots.robot import Robot

from .config_realsense import RealSenseClientConfig


class RealSenseClient(Robot):
    """A client for the Intel RealSense camera, compliant with the LeRobot Robot interface."""

    config_class = RealSenseClientConfig
    name = "realsense_client"

    def __init__(self, config: RealSenseClientConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False

        # Initialize RealSense pipeline
        self.pipeline = rs.pipeline()
        self.rs_config = rs.config()
        self.rs_config.enable_stream(
            rs.stream.color,
            self.config.width,
            self.config.height,
            rs.format.bgr8,
            self.config.fps,
        )

    @property
    def observation_features(self) -> dict:
        return {"observation.image": (self.config.height, self.config.width, 3)}

    @property
    def action_features(self) -> dict:
        # The camera is a sensor and does not have actions.
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError("RealSense camera is already connected.")
        try:
            self.pipeline.start(self.rs_config)
            self._is_connected = True
            logging.info("RealSense camera connected successfully.")
        except RuntimeError as e:
            logging.error(f"Failed to start RealSense pipeline: {e}")
            self._is_connected = False

    @property
    def is_calibrated(self) -> bool:
        # Calibration is not applicable for a standalone camera.
        return True

    def calibrate(self) -> None:
        # No-op, calibration is not needed.
        pass

    def configure(self) -> None:
        # No-op, configuration is handled in __init__.
        pass

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError("RealSense camera is not connected.")

        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                # Return a black image if no frame is available
                image = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            else:
                image = np.asanyarray(color_frame.get_data())
            return {"observation.image": torch.from_numpy(image)}
        except RuntimeError as e:
            logging.error(f"Failed to get frame from RealSense camera: {e}")
            # Return a black image on error
            image = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
            return {"observation.image": torch.from_numpy(image)}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # No-op, the camera does not receive actions.
        return action

    def disconnect(self) -> None:
        if self.is_connected:
            self.pipeline.stop()
            self._is_connected = False
            logging.info("RealSense camera disconnected.")
