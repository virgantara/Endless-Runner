import torch
import torchaudio
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from models import MLPClassifier  # your model

# === Settings ===
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
INPUT_SIZE = 40
NUM_CLASSES = 4
LABELS = ['down', 'left', 'right', 'up']
MODEL_PATH = 'checkpoints/exp/models/best_model.pth'

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === MFCC Transform ===
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=INPUT_SIZE,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": INPUT_SIZE}
)

# === Global variable for volume update ===


# === Audio Recording and Prediction Thread ===
def record_and_predict():
    frames = []
    window_size = int(0.05 * SAMPLE_RATE)  # 50ms chunks
    total_samples = int(DURATION * SAMPLE_RATE)

    def callback(indata, frames_count, time, status):
        rms = np.sqrt(np.mean(indata**2))
        volume_level.set(min(rms * 500, 100))  # scale volume to 0-100
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=window_size, callback=callback):
        sd.sleep(int(DURATION * 1000))

    # Concatenate frames and predict
    audio = np.concatenate(frames, axis=0).T  # shape: [1, N]
    waveform = torch.tensor(audio, dtype=torch.float32)

    if waveform.shape[0] != 1:
        waveform = waveform.unsqueeze(0)

    features = mfcc_transform(waveform)
    features = features.mean(dim=-1).to(device)

    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()
        prediction = LABELS[pred]
        result_var.set(f"Predicted: {prediction}")

def on_button_click():
    result_var.set("Listening...")
    threading.Thread(target=record_and_predict, daemon=True).start()

# === GUI Setup ===
root = tk.Tk()
root.title("🎙 Voice Command Recognition")
volume_level = tk.DoubleVar()

frame = ttk.Frame(root, padding=20)
frame.grid()

title = ttk.Label(frame, text="Voice Command Classifier", font=("Arial", 16))
title.grid(column=0, row=0, pady=10)

button = ttk.Button(frame, text="🎧 Start Listening", command=on_button_click)
button.grid(column=0, row=1, pady=10)

progress = ttk.Progressbar(frame, orient='horizontal', length=300,
                           mode='determinate', maximum=100,
                           variable=volume_level)
progress.grid(column=0, row=2, pady=10)

result_var = tk.StringVar()
result_label = ttk.Label(frame, textvariable=result_var, font=("Arial", 14))
result_label.grid(column=0, row=3, pady=10)

# === Volume Meter Update Loop ===
def update_gui():
    root.after(50, update_gui)  # refresh every 50ms

update_gui()
root.mainloop()
