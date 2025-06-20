import eventlet
eventlet.monkey_patch()

import pickle
import sys
from pathlib import Path

from flask import Flask, render_template
from flask_socketio import SocketIO
from vapi.vapi import Vapi
import zmq

# --- Configuration ---
VAPI_API_KEY = "YOUR_VAPI_API_KEY"  # <-- IMPORTANT: REPLACE WITH YOUR VAPI KEY
ZMQ_PORT = 5555

# --- Flask & SocketIO Setup ---
app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')

# --- Vapi Setup ---
vapi = None
if VAPI_API_KEY != "YOUR_VAPI_API_KEY":
    try:
        vapi = Vapi(api_key=VAPI_API_KEY)
        print("Vapi client initialized.")
    except Exception as e:
        print(f"Failed to initialize Vapi: {e}")
else:
    print("Vapi API key not set. Voice output will be disabled.")

def zmq_listener():
    """Listens to ZMQ messages from the vision process and relays them to clients."""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{ZMQ_PORT}")
    socket.setsockopt_string(zmq.SUBSCRIBE, 'vision')
    print("ZMQ listener connected.")

    while True:
        try:
            _, message = socket.recv_multipart()
            payload = pickle.loads(message)
            event_type = payload.get('type')

            if event_type in ['video_frame', 'raw_data', 'log', 'status']:
                socketio.emit(event_type, payload)
            
            if event_type == 'log' and vapi and payload.get('prediction'):
                prediction_text = payload['prediction'].replace('_', ' ')
                vapi.say(text=prediction_text)

        except Exception as e:
            print(f"Error processing ZMQ message: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send initial status updates
    socketio.emit('status', {'name': 'camera', 'status': 'connected'})
    vapi_status = 'connected' if vapi else 'disconnected'
    socketio.emit('status', {'name': 'vapi', 'status': vapi_status})

if __name__ == '__main__':
    print("--- Dashboard Server Starting ---")
    # Start the ZMQ listener in a background greenlet
    socketio.start_background_task(zmq_listener)
    print("Starting server at http://127.0.0.1:5000")
    socketio.run(app, host='0.0.0.0', port=5000)
