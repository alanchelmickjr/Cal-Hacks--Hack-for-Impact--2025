# Sign Language Communicator: A Cal Hacks Project

## Project Goal

This project aims to create a robotic sign language interpreter. Using a LeRobot arm, a RealSense camera, and cutting-edge AI, we will translate American Sign Language (ASL) gestures into verbal commands and actions. This serves as an assistive technology, empowering individuals who use sign language to interact with smart devices and communicate more seamlessly.

Our focus for this 2-day hackathon is to build a proof-of-concept that recognizes a few key ASL phrases ("bring straw," "read the news," "call mom") and triggers corresponding actions via the Vapi API.

## Tech Stack

*   **Hardware:** LeRobot Arm, Intel RealSense Camera, Raspberry Pi
*   **AI & Vision:** `smolvla` for visual language understanding, Pose Estimation
*   **Language & Voice:** Claude 4 for inference, Vapi for voice commands (TTS, calls, texts)
*   **Core Framework:** `lerobot`

## 2-Day Hackathon Plan (Submit by 4 PM Saturday)

### Day 1: Foundation & Data

-   [ ] **(Fri. Morning) Environment Setup:**
    -   [ ] Configure Raspberry Pi with all necessary dependencies.
    -   [ ] Ensure `lerobot` and Python environment is ready.
    -   [ ] Connect and verify communication with the RealSense camera.
-   [ ] **(Fri. Afternoon) Vision Pipeline:**
    -   [ ] Implement pose estimation to track hand and arm keypoints from the RealSense feed.
    -   [ ] Create a data logging script to capture gesture sequences.
-   [ ] **(Fri. Evening) Dataset Collection:**
    -   [ ] Collect a small, focused dataset for our target ASL signs: "bring straw," "read the news," "call mom."
    -   [ ] Preprocess and label the collected data.

### Day 2: Training, Integration & Submission

-   [ ] **(Sat. Morning) Model Training:**
    -   [ ] Train a `smolvla`-based model on the custom gesture dataset to recognize the target signs.
    -   [ ] Test and validate model accuracy.
-   [ ] **(Sat. Midday) API Integration:**
    -   [ ] Write a script to take the model's output (recognized sign).
    -   [ ] Use Claude 4 to interpret the intent and formulate a command.
    -   [ ] Integrate with the Vapi API to execute actions (e.g., make a call, send a text, or generate a TTS response).
-   [ ] **(Sat. Afternoon) Final Touches & Submission (Deadline: 4 PM):**
    -   [ ] Connect the full pipeline: Camera -> Pose Estimation -> Model -> Vapi.
    -   [ ] Record a compelling demo video.
    -   [ ] Finalize this README and prepare submission package.
    -   [ ] **SUBMIT!**

---
*This project builds upon the [lerobot library](https://github.com/huggingface/lerobot). The original library documentation can be found in [LEROBOT_README.md](./LEROBOT_README.md).*
