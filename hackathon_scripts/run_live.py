import sys
from pathlib import Path
import time

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
from vapi.vapi import Vapi

# --- Setup paths ---
sys.path.append(str(Path(__file__).parent.parent))

from hackathon_scripts.train_model import GestureLSTM, pad_sequences
from lerobot.common.robots.realsense.config_realsense import RealSenseClientConfig
from lerobot.common.robots.realsense.realsense_client import RealSenseClient

# --- Configuration ---
MODEL_PATH = "hackathon_scripts/gesture_model.pth"
ENCODER_PATH = "hackathon_scripts/label_encoder.joblib"
VAPI_API_KEY = "YOUR_VAPI_API_KEY"  # <-- IMPORTANT: REPLACE WITH YOUR VAPI KEY

# --- Model & Inference Parameters ---
INPUT_SIZE = 63
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQUENCE_LENGTH = 75  # The max_len from training, adjust if needed
CONFIDENCE_THRESHOLD = 0.8
PAUSE_THRESHOLD_SECONDS = 1.0  # Seconds of no hand detected to trigger prediction

def main():
    """
    Main function to run the live sign language interpreter.
    """
    if VAPI_API_KEY == "YOUR_VAPI_API_KEY":
        print("ERROR: Please replace 'YOUR_VAPI_API_KEY' in the script with your actual Vapi API key.")
        return

    # --- Load Model and Encoder ---
    print("Loading model and label encoder...")
    label_encoder = joblib.load(ENCODER_PATH)
    NUM_CLASSES = len(label_encoder.classes_)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- Initialize Vapi ---
    vapi = Vapi(api_key=VAPI_API_KEY)

    # --- Initialize MediaPipe and RealSense ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
    mp_drawing = mp.solutions.drawing_utils

    robot = RealSenseClient(RealSenseClientConfig())
    robot.connect()
    if not robot.is_connected:
        print("Failed to connect to camera. Exiting.")
        return

    # --- Main Loop ---
    print("\nStarting live interpreter... Press 'q' to quit.")
    sequence_data = []
    last_hand_detected_time = time.time()
    last_prediction = ""

    try:
        while True:
            obs = robot.get_observation()
            frame_tensor = obs.get("observation.image")
            if frame_tensor is None: continue

            frame = frame_tensor.numpy()
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                last_hand_detected_time = time.time()
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                sequence_data.append(frame_landmarks)
            
            # --- Prediction Logic ---
            time_since_last_detection = time.time() - last_hand_detected_time
            if not hand_detected and time_since_last_detection > PAUSE_THRESHOLD_SECONDS and sequence_data:
                print(f"Gesture ended. Predicting from {len(sequence_data)} frames...")
                
                # Prepare data for model
                padded_seq = pad_sequences([np.array(sequence_data)], SEQUENCE_LENGTH)
                input_tensor = torch.from_numpy(padded_seq).to(device)

                # Get prediction
                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = torch.softmax(output, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)

                if confidence.item() > CONFIDENCE_THRESHOLD:
                    prediction = label_encoder.inverse_transform([predicted_idx.item()])[0]
                    last_prediction = f"{prediction} ({confidence.item():.2f})"
                    print(f"Prediction: {last_prediction}")
                    
                    # Use Vapi to speak the prediction
                    vapi.say(text=prediction.replace('_', ' '))
                else:
                    last_prediction = f"Low confidence ({confidence.item():.2f})"
                    print(last_prediction)

                # Reset for next gesture
                sequence_data = []

            # --- Display Info ---
            cv2.putText(frame, f"Prediction: {last_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frames: {len(sequence_data)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.imshow("Sign Language Interpreter", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        print("\nShutting down...")
        hands.close()
        robot.disconnect()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
