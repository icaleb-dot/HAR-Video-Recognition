"""UCF101 video dataset and DataLoader factory."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import VideoClipTransform


def load_config(path: str | Path = "config/config.yaml") -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def sample_video_frames(video_path: str | Path, num_frames: int) -> torch.Tensor:
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        capture.release()
        raise RuntimeError(f"Video contains no readable frames: {video_path}")
    indices = torch.linspace(0, count - 1, num_frames).round().long().tolist()
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            if not frames:
                capture.release()
                raise RuntimeError(f"Could not read frame {index}: {video_path}")
            frame = frames[-1]
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    capture.release()
    return torch.from_numpy(np.stack(frames))


class UCF101VideoDataset(Dataset):
    def __init__(self, root: str | Path, split: str, classes: list[str], num_frames: int = 16,
                 image_size: int = 112, mean=None, std=None, augment: bool = False):
        self.root = Path(root) / split
        self.class_to_idx = {name: index for index, name in enumerate(classes)}
        self.transform = VideoClipTransform(image_size, mean, std, augment)
        self.num_frames = num_frames
        self.samples = [(path, self.class_to_idx[name]) for name in classes
                        for path in sorted((self.root / name).glob("*.avi"))]
        if not self.samples:
            raise FileNotFoundError(f"No .avi videos found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        clip = self.transform(sample_video_frames(path, self.num_frames))
        return clip, torch.tensor(label, dtype=torch.int64)


def create_dataloaders(config_path: str | Path = "config/config.yaml"):
    cfg = load_config(config_path)
    root = Path(cfg["paths"]["extracted_root"])
    vcfg, tcfg = cfg["video"], cfg["training"]
    loaders = {}
    mapping = {name: idx for idx, name in enumerate(cfg["classes"])}
    generator = torch.Generator().manual_seed(cfg.get("seed", 42))
    for split in ("train", "val", "test"):
        dataset = UCF101VideoDataset(root, split, cfg["classes"], vcfg["num_frames"],
            vcfg["image_size"], vcfg["mean"], vcfg["std"],
            split == "train" and vcfg.get("train_temporal_augmentation", False))
        loaders[split] = DataLoader(dataset, batch_size=tcfg["batch_size"],
            shuffle=split == "train", num_workers=tcfg["num_workers"],
            pin_memory=tcfg.get("pin_memory", False), generator=generator)
    return loaders, mapping
