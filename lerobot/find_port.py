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

"""
Helper to find the USB port associated with your MotorsBus.

Example:

```shell
python -m lerobot.find_port
```
"""

import platform
import time
import argparse
import os
from pathlib import Path


def find_available_ports():
    from serial.tools import list_ports  # Part of pyserial library

    if platform.system() == "Windows":
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_port(set_env: str | None = None, silent: bool = False) -> str:
    if not silent:
        print("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    if not silent:
        print("Ports before disconnecting:", ports_before)

    if not silent:
        print("Remove the USB cable from your MotorsBus and press Enter when done.")
        input()  # Wait for user to disconnect the device
    else:
        # In silent mode we skip the unplug step – useful for automated scripts where only one device is present.
        time.sleep(0.2)

    time.sleep(0.5)  # Allow some time for port to be released
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))

    if len(ports_diff) == 1:
        port = ports_diff[0]
        if not silent:
            print(f"The port of this MotorsBus is '{port}'")
            print("Reconnect the USB cable.")
        if set_env:
            os.environ[set_env] = port
            # Emit a shell-friendly export line so the user can copy-paste.
            print(f"export {set_env}={port}")
    elif len(ports_diff) == 0:
        raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect USB serial port of MotorsBus.")
    parser.add_argument(
        "--env",
        dest="env_var",
        metavar="ENV_VAR",
        help="If provided, set the detected port to this environment variable and print an export line.",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Run without interactive unplug prompt (assumes only one candidate device).",
    )
    args = parser.parse_args()

    find_port(set_env=args.env_var, silent=args.silent)
