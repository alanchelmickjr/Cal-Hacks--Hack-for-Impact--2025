import json
import os
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

# --- Configuration ---
DATA_DIR = "data"
MODEL_SAVE_PATH = "hackathon_scripts/gesture_model.pth"
ENCODER_SAVE_PATH = "hackathon_scripts/label_encoder.joblib"

# --- Hyperparameters ---
INPUT_SIZE = 63  # 21 landmarks * 3 coordinates (x, y, z)
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 0  # Will be determined from the data
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 0.001


class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GestureLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def load_data(data_dir):
    sequences = []
    labels = []
    max_len = 0
    for gesture_name in os.listdir(data_dir):
        gesture_path = Path(data_dir) / gesture_name
        if not gesture_path.is_dir():
            continue
        for file_name in os.listdir(gesture_path):
            if file_name.endswith(".json"):
                with open(gesture_path / file_name, 'r') as f:
                    data = json.load(f)
                    # Flatten landmarks: 21 * [x, y, z] -> 63 features
                    flat_data = [np.array(frame).flatten() for frame in data]
                    sequences.append(np.array(flat_data))
                    labels.append(gesture_name)
                    if len(flat_data) > max_len:
                        max_len = len(flat_data)
    return sequences, labels, max_len


def pad_sequences(sequences, max_len):
    padded_sequences = np.zeros((len(sequences), max_len, INPUT_SIZE), dtype=np.float32)
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq), :] = seq
    return padded_sequences


def main():
    print("Loading data...")
    sequences, labels, max_len = load_data(DATA_DIR)
    if not sequences:
        print("No data found. Please run data_collector.py first.")
        return

    print(f"Found {len(sequences)} sequences from {len(set(labels))} gestures.")
    print(f"Maximum sequence length: {max_len}")

    print("Preprocessing data...")
    padded_sequences = pad_sequences(sequences, max_len)

    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    global NUM_CLASSES
    NUM_CLASSES = len(label_encoder.classes_)

    X_train, X_val, y_train, y_val = train_test_split(
        padded_sequences, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GestureLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for i, (seqs, lbls) in enumerate(train_loader):
            seqs = seqs.to(device)
            lbls = lbls.to(device)

            outputs = model(seqs)
            loss = criterion(outputs, lbls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for seqs, lbls in val_loader:
                seqs = seqs.to(device)
                lbls = lbls.to(device)
                outputs = model(seqs)
                _, predicted = torch.max(outputs.data, 1)
                total += lbls.size(0)
                correct += (predicted == lbls).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}, Validation Accuracy: {accuracy:.2f}%')

    print("Training finished.")

    print(f"Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Done.")

if __name__ == '__main__':
    main()
