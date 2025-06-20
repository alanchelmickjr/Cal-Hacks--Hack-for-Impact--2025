import base64
import pickle
import sys
import time
from pathlib import Path

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import zmq

# --- Setup paths ---
sys.path.append(str(Path(__file__).parent.parent.parent))

from hackathon_scripts.train_model import GestureLSTM, pad_sequences
from lerobot.common.robots.realsense.config_realsense import RealSenseClientConfig
try:
    from lerobot.common.robots.realsense.realsense_client import RealSenseClient
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

# --- Configuration ---
MODEL_PATH = "hackathon_scripts/gesture_model.pth"
ENCODER_PATH = "hackathon_scripts/label_encoder.joblib"
ZMQ_PORT = 5555

# --- Model & Inference Parameters ---
INPUT_SIZE = 63
HIDDEN_SIZE = 128
NUM_LAYERS = 2
SEQUENCE_LENGTH = 75
CONFIDENCE_THRESHOLD = 0.8
PAUSE_THRESHOLD_SECONDS = 1.0

def main():
    """Main processing loop for camera, AI, and ZMQ publishing."""
    print("--- Vision Process Starting ---")

    # --- ZMQ Setup ---
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ZMQ_PORT}")
    print(f"ZMQ publisher running on port {ZMQ_PORT}")

    # --- Load Model ---
    print("Loading model...")
    label_encoder = joblib.load(ENCODER_PATH)
    NUM_CLASSES = len(label_encoder.classes_)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")

    # --- Initialize MediaPipe and RealSense ---
    print("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5, max_num_hands=1)
    print("MediaPipe initialized.")

    robot = None
    if REALSENSE_AVAILABLE:
        try:
            print("Initializing RealSenseClient...")
            robot = RealSenseClient(RealSenseClientConfig())
            print("RealSenseClient object created. Connecting...")
            robot.connect()
            if not robot.is_connected:
                print("WARN: Could not connect to RealSense camera. Starting in mock mode.")
                robot = None
            else:
                print("RealSense camera connected successfully.")
        except Exception as e:
            print(f"WARN: An exception occurred during RealSense init: {e}. Starting in mock mode.")
            robot = None
    else:
        print("WARN: pyrealsense2 library not found. Starting in mock mode.")

    try:
        if robot is None:
            # --- Mock Mode Loop ---
            print("Running in MOCK DATA mode.")
            while True:
                # Send status update
                status_payload = {'type': 'status', 'name': 'camera', 'status': 'disconnected'}
                socket.send_multipart([b'vision', pickle.dumps(status_payload)])

                # Create a placeholder frame
                mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(mock_frame, "Camera Not Found", (160, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                _, buffer = cv2.imencode('.jpg', mock_frame)
                b64_frame = base64.b64encode(buffer).decode('utf-8')
                video_payload = {'type': 'video_frame', 'image': b64_frame}
                socket.send_multipart([b'vision', pickle.dumps(video_payload)])
                
                time.sleep(1) # Send mock data every second
        else:
            # --- Real Camera Loop ---
            print("Running in REAL CAMERA mode.")
            sequence_data = []
            last_hand_detected_time = time.time()
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
                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    frame_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    sequence_data.append(frame_landmarks)
                    
                    raw_data_payload = {'type': 'raw_data', 'data': [f'{coord:.3f}' for coord in frame_landmarks]}
                    socket.send_multipart([b'vision', pickle.dumps(raw_data_payload)])

                if not hand_detected and time.time() - last_hand_detected_time > PAUSE_THRESHOLD_SECONDS and sequence_data:
                    padded_seq = pad_sequences([np.array(sequence_data)], SEQUENCE_LENGTH)
                    input_tensor = torch.from_numpy(padded_seq).to(device)

                    with torch.no_grad():
                        output = model(input_tensor)
                        probabilities = torch.softmax(output, dim=1)
                        confidence, predicted_idx = torch.max(probabilities, 1)

                    if confidence.item() > CONFIDENCE_THRESHOLD:
                        prediction = label_encoder.inverse_transform([predicted_idx.item()])[0]
                        log_msg = f'Prediction: {prediction} (Conf: {confidence.item():.2f})'
                        log_payload = {'type': 'log', 'data': log_msg, 'prediction': prediction}
                    else:
                        log_msg = f'Low confidence: {confidence.item():.2f}'
                        log_payload = {'type': 'log', 'data': log_msg, 'prediction': None}
                    
                    socket.send_multipart([b'vision', pickle.dumps(log_payload)])
                    sequence_data = []

                _, buffer = cv2.imencode('.jpg', frame)
                b64_frame = base64.b64encode(buffer).decode('utf-8')
                video_payload = {'type': 'video_frame', 'image': b64_frame}
                socket.send_multipart([b'vision', pickle.dumps(video_payload)])
                
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nShutting down vision process...")
    finally:
        socket.close()
        context.term()
        hands.close()
        if robot:
            robot.disconnect()
        print("Vision process stopped.")

if __name__ == "__main__":
    main()
