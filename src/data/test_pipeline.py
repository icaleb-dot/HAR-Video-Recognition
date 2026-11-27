"""Quick smoke test for the shared dataset and DataLoader contract."""
import argparse

import torch

from src.data.dataset import create_dataloaders, load_config


def check_pipeline(config_path="config/config.yaml"):
    config = load_config(config_path)
    loaders, class_to_idx = create_dataloaders(config_path)
    assert set(loaders) == {"train", "val", "test"}
    assert list(class_to_idx) == config["classes"]
    video = config["video"]
    expected_clip_shape = (3, video["num_frames"], video["image_size"], video["image_size"])
    for split, loader in loaders.items():
        clips, labels = next(iter(loader))
        assert clips.ndim == 5 and tuple(clips.shape[1:]) == expected_clip_shape, (split, clips.shape)
        assert clips.dtype == torch.float32, (split, clips.dtype)
        assert labels.ndim == 1 and labels.dtype == torch.int64, (split, labels.shape, labels.dtype)
        assert labels.min().item() >= 0 and labels.max().item() < len(class_to_idx)
        print(f"{split}: clips={tuple(clips.shape)} {clips.dtype}; labels={tuple(labels.shape)} {labels.dtype}")
    print(f"Contrato de datos correcto para {len(class_to_idx)} clases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    check_pipeline(parser.parse_args().config)
