import os
import torch
from torch.utils.data import Dataset
import torchaudio

class VoiceCommandDataset(Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, class_name = self.samples[idx]
        label_idx = self.class_to_idx[class_name]  # <- integer class index

        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label_idx  # <- NOT one-hot anymore
