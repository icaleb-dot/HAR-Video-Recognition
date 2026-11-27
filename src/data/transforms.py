"""Spatial preprocessing and lightweight temporal augmentation for video clips."""
import random

import torch
import torch.nn.functional as F


class VideoClipTransform:
    def __init__(self, image_size=112, mean=None, std=None, augment=False):
        self.image_size = image_size
        self.mean = torch.tensor(mean or [0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1)
        self.std = torch.tensor(std or [0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1)
        self.augment = augment

    def __call__(self, frames):
        # Input is (T,H,W,C), uint8 RGB. Output is (C,T,H,W), float32.
        clip = frames.permute(0, 3, 1, 2).float() / 255.0
        if self.augment and random.random() < 0.5:
            clip = clip.flip(0)
        clip = F.interpolate(clip, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        clip = (clip - self.mean) / self.std
        return clip.permute(1, 0, 2, 3).contiguous()
