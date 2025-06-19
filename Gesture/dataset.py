import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_video
from torchvision import models
import cv2
import numpy as np
from torchvision.transforms import ToTensor

class VideoDatasetCV(Dataset):
    def __init__(self, root_dir, frames_per_clip=16, resize=(112, 112)):
        self.root_dir = root_dir
        self.frames_per_clip = frames_per_clip
        self.resize = resize
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for cls in self.classes:
            folder = os.path.join(self.root_dir, cls)
            for f in os.listdir(folder):
                if f.endswith(('.mp4', '.avi', '.mov')):
                    samples.append((os.path.join(folder, f), self.class_to_idx[cls]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._read_video_opencv(path)
        total = len(frames)

        if total >= self.frames_per_clip:
            start = np.random.randint(0, total - self.frames_per_clip + 1)
            clip = frames[start:start+self.frames_per_clip]
        else:
            pad = [frames[-1]] * (self.frames_per_clip - total)
            clip = frames + pad

        clip_tensor = torch.stack([ToTensor()(f) for f in clip])  # [T, C, H, W]
        return clip_tensor, label

    def _read_video_opencv(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            frames.append(frame)
        cap.release()
        return frames