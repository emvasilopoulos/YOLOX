#!/usr/bin/env python3
"""
Convert OpenImages bbox annotations to COCO format, keeping only classes that exist in COCO.

Inputs:
  1) COCO classes YAML (Ultralytics-style):
        names:
          0: person
          1: bicycle
          ...

  2) OpenImages class descriptions CSV:
        LabelName,DisplayName
        /m/0mkg,Accordion
        ...

  3) OpenImages bbox annotations CSV:
        ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside
        0001...,xclick,/m/...,1,0.02,0.96,0.07,0.80,0,0,0,0,0
        ...

Optional (recommended for pixel-accurate COCO bboxes):
  4) Image sizes CSV with at least these columns:
        ImageID,Width,Height
     If not provided, bboxes remain normalized and images get width=height=1 (COCO-style JSON,
     but many consumers expect pixel units).

Example:
  python oi_to_coco.py \
    --coco_yaml coco_names.yaml \
    --oi_class_csv class-descriptions-boxable.csv \
    --oi_bbox_csv train-annotations-bbox.csv \
    --oi_image_sizes_csv image_sizes.csv \
    --output_json out_coco.json
"""

import argparse
import csv
import json
import re
from typing import Dict, Tuple

import yaml

# A small synonym layer to improve matching between COCO class names and OpenImages DisplayName.
# Keys are COCO names (lowercase), values are candidate OpenImages DisplayName strings (lowercase).
COCO_TO_OI_SYNONYMS = {
    "airplane": ["aircraft", "aeroplane", "airplane"],
    "tv": ["television", "tv", "monitor"],
    "cell phone": ["mobile phone", "cell phone", "telephone"],
    "couch": ["sofa", "couch"],
    "dining table": ["dining table", "table"],
    "potted plant": ["houseplant", "potted plant", "plant"],
    "motorcycle": ["motorcycle", "motorbike"],
    "remote": ["remote control"],
    "sports ball": ["ball", "sports ball"],
    "traffic light": ["traffic light"],
    "fire hydrant": ["fire hydrant"],
    "stop sign": ["stop sign"],
    "parking meter": ["parking meter"],
    "tennis racket": ["tennis racket", "tennis racquet"],
    "handbag": ["handbag", "purse"],
    "backpack": ["backpack", "rucksack"],
    "wine glass": ["wine glass"],
    "hot dog": ["hot dog"],
    "baseball bat": ["baseball bat"],
    "baseball glove": ["baseball glove"],
    "toothbrush": ["toothbrush"],
    "hair drier": ["hair dryer", "hair drier"],
}


def norm_name(s: str) -> str:
    """Normalize a class name for robust matching."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def load_coco_yaml(path: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    if not isinstance(y, dict) or "names" not in y:
        raise ValueError("COCO YAML must have top-level key 'names'.")

    names = y["names"]
    if not isinstance(names, dict):
        raise ValueError(
            "'names' must be a mapping like {0: person, 1: bicycle, ...}.")

    out: Dict[int, str] = {}
    for k, v in names.items():
        # YAML might parse keys as int or str
        idx = int(k)
        out[idx] = str(v)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def load_oi_class_descriptions(path: str) -> Dict[str, str]:
    """LabelName -> DisplayName"""
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"LabelName", "DisplayName"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"OpenImages class CSV must contain columns: {sorted(required)}"
            )
        for row in reader:
            mapping[row["LabelName"]] = row["DisplayName"]
    return mapping


def load_image_sizes(path: str) -> Dict[str, Tuple[int, int]]:
    """ImageID -> (width, height)"""
    sizes: Dict[str, Tuple[int, int]] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"ImageID", "Width", "Height"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                f"Image sizes CSV must contain columns: {sorted(required)}")
        for row in reader:
            image_id = row["ImageID"]
            w = int(float(row["Width"]))
            h = int(float(row["Height"]))
            sizes[image_id] = (w, h)
    return sizes


def build_labelname_to_coco_id(
    coco_id_to_name: Dict[int, str],
    oi_label_to_display: Dict[str, str],
) -> Dict[str, int]:
    """
    Create mapping from OpenImages LabelName to COCO category_id by matching DisplayName to COCO names.
    Matching strategy:
      - exact normalized match
      - then synonym match via COCO_TO_OI_SYNONYMS
    """
    coco_name_to_id = {norm_name(n): cid for cid, n in coco_id_to_name.items()}

    # Build reverse index for OpenImages DisplayName -> LabelName(s)
    oi_disp_to_labels: Dict[str, list] = {}
    for lbl, disp in oi_label_to_display.items():
        oi_disp_to_labels.setdefault(norm_name(disp), []).append(lbl)

    label_to_coco: Dict[str, int] = {}

    # 1) exact matches
    for coco_norm, coco_id in coco_name_to_id.items():
        if coco_norm in oi_disp_to_labels:
            for lbl in oi_disp_to_labels[coco_norm]:
                label_to_coco[lbl] = coco_id

    # 2) synonym matches (only fill if not already mapped)
    for coco_norm, coco_id in coco_name_to_id.items():
        syns = COCO_TO_OI_SYNONYMS.get(coco_norm, [])
        for syn in syns:
            syn_norm = norm_name(syn)
            if syn_norm in oi_disp_to_labels:
                for lbl in oi_disp_to_labels[syn_norm]:
                    label_to_coco.setdefault(lbl, coco_id)

    return label_to_coco


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_yaml",
                    required=True,
                    help="YAML with COCO class names under 'names:'")
    ap.add_argument("--oi_class_csv",
                    required=True,
                    help="OpenImages class descriptions CSV")
    ap.add_argument("--oi_bbox_csv",
                    required=True,
                    help="OpenImages bbox annotations CSV")
    ap.add_argument("--oi_image_sizes_csv",
                    required=True,
                    help="CSV: ImageID,Width,Height")
    ap.add_argument("--output_json",
                    required=True,
                    help="Output COCO-format JSON file")
    ap.add_argument("--min_confidence",
                    type=float,
                    default=0.0,
                    help="Filter bboxes by Confidence >= this")
    args = ap.parse_args()

    coco_id_to_name = load_coco_yaml(args.coco_yaml)
    oi_label_to_display = load_oi_class_descriptions(args.oi_class_csv)
    image_sizes = load_image_sizes(args.oi_image_sizes_csv)

    label_to_coco_id = build_labelname_to_coco_id(coco_id_to_name,
                                                  oi_label_to_display)
    keep_labels = set(label_to_coco_id.keys())

    # Prepare COCO categories (keep IDs as given in YAML)
    categories = [{
        "id": cid,
        "name": name
    } for cid, name in coco_id_to_name.items()]

    images = []
    annotations = []

    oi_imageid_to_coco_imageid: Dict[str, int] = {}
    next_image_id = 1
    next_ann_id = 1

    with open(args.oi_bbox_csv, "r", encoding="utf-8", newline="") as f:
        bbox_csv_reader = csv.DictReader(f)
        required = {
            "ImageID", "LabelName", "Confidence", "XMin", "XMax", "YMin",
            "YMax", "IsGroupOf"
        }
        if not required.issubset(set(bbox_csv_reader.fieldnames or [])):
            raise ValueError(
                f"OpenImages bbox CSV must contain columns at least: {sorted(required)}"
            )

        for row in bbox_csv_reader:
            label = row["LabelName"]
            if label not in keep_labels:
                continue

            conf = float(row["Confidence"])
            if conf < args.min_confidence:
                continue

            image_id_str = row["ImageID"]

            if image_id_str not in oi_imageid_to_coco_imageid:
                coco_img_id = next_image_id
                next_image_id += 1
                oi_imageid_to_coco_imageid[image_id_str] = coco_img_id

                if image_sizes and image_id_str in image_sizes:
                    w, h = image_sizes[image_id_str]
                else:
                    # TODO - might skip image
                    raise ValueError(
                        f"Image sizes CSV is required for pixel-accurate bboxes, but image ID {image_id_str} not found."
                    )

                images.append({
                    "id": coco_img_id,
                    "file_name": f"{image_id_str}.jpg",
                    "width": w,
                    "height": h,
                })

            coco_img_id = oi_imageid_to_coco_imageid[image_id_str]
            coco_cat_id = label_to_coco_id[label]

            x_min = float(row["XMin"])
            x_max = float(row["XMax"])
            y_min = float(row["YMin"])
            y_max = float(row["YMax"])

            # Clamp to [0,1] defensively
            x_min = max(0.0, min(1.0, x_min))
            x_max = max(0.0, min(1.0, x_max))
            y_min = max(0.0, min(1.0, y_min))
            y_max = max(0.0, min(1.0, y_max))

            if images and images[-1][
                    "id"] == coco_img_id:  # safety check, to avoid buggy data
                img_entry = images[-1]
            else:
                img_entry = None

            if img_entry is None:
                # Find image entry (rare if not appended last due to ordering)
                for im in images:
                    if im["id"] == coco_img_id:
                        img_entry = im
                        break
            assert img_entry is not None

            # de-scale bbox
            W, H = img_entry["width"], img_entry["height"]
            x = x_min * W
            y = y_min * H
            bw = (x_max - x_min) * W
            bh = (y_max - y_min) * H

            # Some OpenImages boxes can be degenerate; skip invalid
            if bw <= 0 or bh <= 0:
                continue

            is_group = int(row.get("IsGroupOf", "0") or "0")
            iscrowd = 1 if is_group == 1 else 0

            annotations.append({
                "id": next_ann_id,
                "image_id": coco_img_id,
                "category_id": coco_cat_id,
                "bbox": [x, y, bw, bh],
                "area": bw * bh,
                "iscrowd": iscrowd
            })
            next_ann_id += 1

    coco_out = {
        "info": {
            "description":
            "Converted from OpenImages to COCO format (filtered to COCO classes)",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(coco_out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output_json}")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories (from COCO YAML): {len(categories)}")
    if not args.oi_image_sizes_csv:
        print(
            "WARNING: No image sizes CSV provided; bboxes are effectively normalized (width=height=1)."
        )


if __name__ == "__main__":
    main()
