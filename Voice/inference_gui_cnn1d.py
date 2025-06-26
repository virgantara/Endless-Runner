import torch
import torchaudio
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from models import CNN1DSoundClassifier  # Ganti jika nama model berbeda

# === Settings ===
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
INPUT_SIZE = 40
NUM_CLASSES = 4
LABELS = ['down', 'left', 'right', 'up']
MODEL_PATH = 'checkpoints/exp_cnn1d/models/best_model.pth'

# === Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1DSoundClassifier(input_channels=INPUT_SIZE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === MFCC Transform ===
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=INPUT_SIZE,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": INPUT_SIZE}
)


# === Inference Function ===
def record_and_predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    window_size = int(0.05 * SAMPLE_RATE)  # 50ms chunks

    def callback(indata, frames_count, time, status):
        rms = np.sqrt(np.mean(indata**2))
        volume_level.set(min(rms * 500, 100))  # scale to 0–100
        frames.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                        blocksize=window_size, callback=callback):
        sd.sleep(int(DURATION * 1000))

    # Preprocess audio
    audio = np.concatenate(frames, axis=0).T   # shape: [1, N]
    waveform = torch.tensor(audio, dtype=torch.float32)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    features = mfcc_transform(waveform)  # [1, 40, T]
    features = features.squeeze(0).unsqueeze(0)  # shape: [1, 40, T]
    features = features.to(device)
    # print("features: ",features.size())
    with torch.no_grad():
        
        output = model(features)
        
        probs = torch.softmax(output, dim=1)
        max_prob, pred = torch.max(probs, dim=1)

        if max_prob.item() < 0.5:
            result_var.set("Uncertain ")
        else:
            prediction = LABELS[pred.item()]
            result_var.set(f"Predicted: {prediction} ({max_prob.item():.2f}) ")

def on_button_click():
    result_var.set("🎤 Listening...")
    threading.Thread(target=record_and_predict, daemon=True).start()

# === GUI Setup ===
root = tk.Tk()
root.title("🎙 Voice Command CNN1D")

volume_level = tk.DoubleVar()

frame = ttk.Frame(root, padding=20)
frame.grid()

title = ttk.Label(frame, text="Voice Command Classifier (CNN1D)", font=("Arial", 16))
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

# Volume Meter Refresh
def update_gui():
    root.after(50, update_gui)

update_gui()
root.mainloop()
