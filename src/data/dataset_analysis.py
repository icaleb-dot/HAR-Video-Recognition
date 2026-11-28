"""Create per-video metadata, class counts, and a class-distribution figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import pandas as pd

from src.data.dataset import load_config


def analyze_dataset(config_path: str | Path = "config/config.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    root = Path(config["paths"]["extracted_root"])
    classes = config["classes"]
    records = []
    for split in ("train", "val", "test"):
        for class_name in classes:
            videos = sorted((root / split / class_name).glob("*.avi"))
            for video_path in videos:
                cap = cv2.VideoCapture(str(video_path))
                if not cap.isOpened():
                    print(f"[WARN] No se pudo abrir: {video_path}")
                    cap.release()
                    continue
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                records.append({"split": split, "class": class_name, "video": video_path.name,
                    "path": str(video_path), "frame_count": frames, "fps": fps,
                    "duration_seconds": frames / fps if fps > 0 else 0.0,
                    "width": width, "height": height, "resolution": f"{width}x{height}"})
    if not records:
        raise FileNotFoundError(f"No se encontraron videos AVI bajo {root}; extrae el TAR primero.")
    return pd.DataFrame(records)


def generate_outputs(frame: pd.DataFrame, results_dir: str | Path = "results") -> None:
    metrics_dir, figures_dir = Path(results_dir) / "metrics", Path(results_dir) / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(metrics_dir / "dataset_metadata.csv", index=False)
    counts = frame.groupby(["split", "class"]).size().unstack(fill_value=0)
    counts.to_csv(metrics_dir / "class_distribution.csv")
    counts.plot(kind="bar", figsize=(13, 6))
    plt.title("UCF101 subset: videos por clase y split")
    plt.xlabel("Actividad")
    plt.ylabel("Número de videos")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_distribution.png", dpi=150)
    plt.close()
    print(f"Videos analizados: {len(frame)}")
    print(counts.to_string())
    print(f"Resultados guardados en {metrics_dir} y {figures_dir}")


def main(config_path: str | Path = "config/config.yaml") -> None:
    config = load_config(config_path)
    frame = analyze_dataset(config_path)
    generate_outputs(frame, config["paths"].get("results", "results"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    main(parser.parse_args().config)
