import torch
import torchaudio
import sounddevice as sd
import numpy as np
from models import MLPClassifier
import time

# === Settings ===
SAMPLE_RATE = 16000
DURATION = 1  # in seconds
NUM_CLASSES = 4
INPUT_SIZE = 40  # MFCC feature size
LABELS = ['down', 'left', 'right', 'up']  # adjust if needed
MODEL_PATH = 'checkpoints/exp/models/best_model.pth'

# === Load model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLPClassifier(input_size=INPUT_SIZE, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# === MFCC transform ===
mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=INPUT_SIZE,
    melkwargs={"n_fft": 400, "hop_length": 160, "n_mels": INPUT_SIZE}
)

def record_audio(duration=1, fs=16000):
    print("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return torch.tensor(audio.T, dtype=torch.float32)

def predict_command(audio_waveform):
    if audio_waveform.shape[0] != 1:
        audio_waveform = audio_waveform.unsqueeze(0)

    features = mfcc_transform(audio_waveform)
    features = features.mean(dim=-1)  # [1, 40]
    features = features.to(device)

    with torch.no_grad():
        output = model(features)
        pred = torch.argmax(output, dim=1).item()
        return LABELS[pred]

if __name__ == "__main__":
    print("Real-time Voice Command Inference (Press Ctrl+C to exit)")
    try:
        while True:
            waveform = record_audio(duration=DURATION, fs=SAMPLE_RATE)
            command = predict_command(waveform)
            print(f">>> Predicted Command: {command}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopped by user.")
