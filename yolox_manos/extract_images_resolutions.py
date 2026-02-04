import csv
import os
from pathlib import Path
from PIL import Image


def extract_image_resolutions(input_dir: str, output_csv: str) -> None:
    input_path = Path(input_dir)
    rows = [("ImageId", "Width", "Height")]

    for file_path in sorted(input_path.iterdir()):
        if not file_path.is_file():
            continue
        try:
            with Image.open(file_path) as img:
                width, height = img.size
            image_id = file_path.stem
            rows.append((image_id, width, height))
        except Exception:
            continue

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


if __name__ == "__main__":
    input_directory = "/home/manos/tools/YOLOX/datasets/OpenImages/validation/"
    output_csv_path = "validation-image-resolutions.csv"
    extract_image_resolutions(input_directory, output_csv_path)
