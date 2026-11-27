
import cv2
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path



DATASET_PATH = Path("data/processed/UCF101_subset")

SPLITS = ["train", "val", "test"]

VIDEO_EXTENSIONS = [".avi", ".mp4", ".mov"]



def analyze_video(video_path):
    """
    Extract basic metadata from a video.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "fps": fps,
        "frame_count": int(frame_count),
        "duration": duration,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}"
    }



def analyze_dataset():

    records = []

    print("\nStarting dataset analysis...\n")

    for split in SPLITS:

        split_path = DATASET_PATH / split

        if not split_path.exists():
            print(f"[WARNING] Missing split: {split}")
            continue

        for class_path in sorted(split_path.iterdir()):

            if not class_path.is_dir():
                continue

            class_name = class_path.name

            videos = [
                path for path in class_path.iterdir()
                if path.suffix.lower() in VIDEO_EXTENSIONS
            ]

            print(
                f"Processing {split}/{class_name}: "
                f"{len(videos)} videos"
            )

            for video_path in videos:

                metadata = analyze_video(video_path)

                if metadata is None:
                    print(f"[ERROR] Cannot open: {video_path}")
                    continue

                records.append({
                    "split": split,
                    "class": class_name,
                    "video": video_path.name,
                    **metadata
                })

    return pd.DataFrame(records)



def generate_statistics(df):

    if df.empty:
        print("No valid videos found.")
        return

    print("\n========== DATASET SUMMARY ==========")

    print(f"\nTotal videos: {len(df)}")

    print("\nVideos per split:")
    print(df["split"].value_counts())

    print("\nVideos per class:")
    print(df.groupby(["split", "class"]).size())

    print("\nDuration statistics:")
    print(df["duration"].describe())

    print("\nResolution distribution:")
    print(df["resolution"].value_counts())

    print("\nFPS distribution:")
    print(df["fps"].describe())



def plot_class_distribution(df):

    if df.empty:
        return

    distribution = (
        df.groupby(["class", "split"])
        .size()
        .unstack(fill_value=0)
    )

    distribution.plot(
        kind="bar",
        figsize=(14, 6)
    )

    plt.title("UCF101 Subset - Class Distribution")
    plt.xlabel("Activity")
    plt.ylabel("Number of videos")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = Path("results/figures")

    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path / "class_distribution.png")

    plt.show()


if __name__ == "__main__":

    df = analyze_dataset()

    generate_statistics(df)

    plot_class_distribution(df)

    output_path = Path("results/metrics")
    output_path.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_path / "dataset_metadata.csv",
        index=False
    )

    print("\nAnalysis completed successfully.")
