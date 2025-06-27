import torch
import torchaudio
import sounddevice as sd
import numpy as np
import tkinter as tk
from tkinter import ttk
import threading
from models import CNN1DSoundClassifier  # Ganti jika nama model berbeda
import socket

TCP_HOST = '127.0.0.1'  # Ganti dengan IP server C# jika perlu
TCP_PORT = 5005

socketClient = None

# === Settings ===
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
INPUT_SIZE = 40
NUM_CLASSES = 6
LABELS = ['down', 'go', 'left', 'right', 'stop', 'up']
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

def send_tcp_command(command):
    try:
        socketClient.sendall(command.encode('utf-8'))
        print(f"[TCP] Sent: {command}")
    except Exception as e:
        print(f"[TCP] Error: {e}")

is_listening = False


def continuous_listen():
    global is_listening
    window_size = int(0.05 * SAMPLE_RATE)  # 50ms

    while is_listening:
        frames = []

        def callback(indata, frames_count, time, status):
            rms = np.sqrt(np.mean(indata**2))
            volume_level.set(min(rms * 500, 100))  # scale
            frames.append(indata.copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32',
                            blocksize=window_size, callback=callback):
            sd.sleep(int(DURATION * 1000))

        audio = np.concatenate(frames, axis=0).T
        global_rms = np.sqrt(np.mean(audio**2))

        if global_rms < 0.05:
            result_var.set("No sound detected")
            continue

        waveform = torch.tensor(audio, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        
        features = mfcc_transform(waveform)  # [1, 40, T]
        features = features.squeeze(0).unsqueeze(0)  # shape: [1, 40, T]
        features = features.to(device)

        with torch.no_grad():
            output = model(features)
            probs = torch.softmax(output, dim=1)
            max_prob, pred = torch.max(probs, dim=1)

            if max_prob.item() < 0.5:
                result_var.set(f"Uncertain ({global_rms:.4f})")
            else:
                prediction = LABELS[pred.item()]
                result_var.set(f"Predicted: {prediction} ({max_prob.item():.2f})")
                send_tcp_command(prediction)


def start_listening():

    global is_listening, socketClient

    if not is_listening:    
        try:
            socketClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            socketClient.connect((TCP_HOST, TCP_PORT))
            print("TCP Connected")
        except Exception as e:
            result_var.set(f"TCP Connection failed: {e}")
            return


        is_listening = True
        result_var.set(" Listening...")

        threading.Thread(target=continuous_listen, daemon=True).start()

def stop_listening():
    global is_listening, socketClient
    is_listening = False
    
    if socketClient:
        try:
            socketClient.close()
            print("[TCPT] closed")
        except Exception as e:
            pass

    socketClient = None
    result_var.set(" Stopped Listening.")

# === Inference Function ===

root = tk.Tk()
root.title("Voice Command CNN1D")
root.configure(bg="#2e2e2e")  # Abu-abu tua

volume_level = tk.DoubleVar()
result_var = tk.StringVar()

style = ttk.Style()
style.theme_use("clam")  # gunakan tema yang bisa dimodifikasi
style.configure("TFrame", background="#2e2e2e")
style.configure("TLabel", background="#2e2e2e", foreground="white", font=("Arial", 12))
style.configure("TButton", background="#444", foreground="white")
style.map("TButton", background=[("active", "#666")])
style.configure("TProgressbar", background="#4caf50")

frame = ttk.Frame(root, padding=20)
frame.grid()

title = ttk.Label(frame, text="Voice Command Classifier (CNN1D)", font=("Arial", 16))
title.grid(column=0, row=0, columnspan=2, pady=10)

start_btn = ttk.Button(frame, text="Start Listening", command=start_listening)
start_btn.grid(column=0, row=1, pady=10)

stop_btn = ttk.Button(frame, text="Stop Listening", command=stop_listening)
stop_btn.grid(column=1, row=1, pady=10)

progress = ttk.Progressbar(frame, orient='horizontal', length=300,
                           mode='determinate', maximum=100,
                           variable=volume_level)
progress.grid(column=0, row=2, columnspan=2, pady=10)

result_label = ttk.Label(frame, textvariable=result_var, font=("Arial", 14))
result_label.grid(column=0, row=3, columnspan=2, pady=10)

# Refresh UI loop
def update_gui():
    root.after(50, update_gui)
update_gui()

root.mainloop()