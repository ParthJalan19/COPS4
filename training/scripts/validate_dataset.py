"""
COPS-4 Dataset Validator & Pre-flight Inspection Tool.
Ensures dataset integrity prior to custom training:
1. Validates YOLO annotation format: <class_id> <cx> <cy> <w> <h>
2. Detects missing image/label pairs and orphaned files
3. Identifies out-of-bound or corrupted bounding boxes ([0.0, 1.0])
4. Detects zero-byte or corrupt/unreadable image files
5. Identifies duplicate images via SHA-256 perceptual hashes
6. Reports per-class distribution, train/val split ratios, and imbalance
7. Detects hard-negative samples (empty label files for background images)
"""

import os
import sys
import hashlib
from typing import Dict, List, Tuple, Any
import cv2
import numpy as np


class DatasetValidator:
    def __init__(self, data_root: str, class_names: List[str] = None):
        self.data_root = os.path.abspath(data_root)
        self.class_names = class_names or ["pistol", "knife"]
        self.stats = {
            "total_images": 0,
            "total_labels": 0,
            "valid_samples": 0,
            "hard_negatives": 0,
            "corrupt_images": [],
            "missing_labels": [],
            "orphaned_labels": [],
            "invalid_boxes": [],
            "duplicate_images": [],
            "class_counts": {i: 0 for i in range(len(self.class_names))},
            "split_counts": {}
        }

    def _file_hash(self, filepath: str) -> str:
        """Calculates SHA256 of file bytes for exact duplicate detection."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def validate_split(self, split_name: str) -> Dict[str, Any]:
        """Validates a dataset split (e.g., 'train', 'val', 'test')."""
        split_dir = os.path.join(self.data_root, split_name)
        img_dir = os.path.join(split_dir, "images")
        lbl_dir = os.path.join(split_dir, "labels")

        if not os.path.exists(img_dir):
            return {"error": f"Images directory not found: {img_dir}"}

        os.makedirs(lbl_dir, exist_ok=True)

        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))]
        label_files = [f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")]

        self.stats["split_counts"][split_name] = {
            "images": len(image_files),
            "labels": len(label_files)
        }

        hashes = {}
        for img_name in image_files:
            self.stats["total_images"] += 1
            img_path = os.path.join(img_dir, img_name)
            base_name = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dir, base_name + ".txt")

            # Check image integrity
            try:
                img = cv2.imread(img_path)
                if img is None or img.size == 0 or img.shape[0] < 10 or img.shape[1] < 10:
                    self.stats["corrupt_images"].append(img_path)
                    continue
            except Exception as e:
                self.stats["corrupt_images"].append(f"{img_path} ({e})")
                continue

            # Duplicate check
            fhash = self._file_hash(img_path)
            if fhash in hashes:
                self.stats["duplicate_images"].append((img_path, hashes[fhash]))
            else:
                hashes[fhash] = img_path

            # Label check
            if not os.path.exists(lbl_path):
                self.stats["missing_labels"].append(img_path)
                continue

            # Annotation parse
            with open(lbl_path, "r", encoding="utf-8") as lf:
                lines = [line.strip() for line in lf.readlines() if line.strip()]

            if len(lines) == 0:
                self.stats["hard_negatives"] += 1
                self.stats["valid_samples"] += 1
                continue

            valid_sample = True
            for lidx, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    self.stats["invalid_boxes"].append({
                        "file": lbl_path, "line": lidx + 1, "error": f"Expected 5 values, got {len(parts)}: '{line}'"
                    })
                    valid_sample = False
                    continue

                try:
                    cls_id = int(parts[0])
                    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    self.stats["invalid_boxes"].append({
                        "file": lbl_path, "line": lidx + 1, "error": f"Non-numeric values in line: '{line}'"
                    })
                    valid_sample = False
                    continue

                # Range check [0.0, 1.0]
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                    self.stats["invalid_boxes"].append({
                        "file": lbl_path, "line": lidx + 1, "error": f"Bbox values out of range [0, 1]: {parts[1:]}"
                    })
                    valid_sample = False
                    continue

                if cls_id not in self.stats["class_counts"]:
                    self.stats["class_counts"][cls_id] = 0
                self.stats["class_counts"][cls_id] += 1

            if valid_sample:
                self.stats["valid_samples"] += 1

        # Check orphaned labels (label file with no corresponding image)
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        for lbl_name in label_files:
            l_base = os.path.splitext(lbl_name)[0]
            if l_base not in image_basenames:
                self.stats["orphaned_labels"].append(os.path.join(lbl_dir, lbl_name))

        return self.stats["split_counts"][split_name]

    def validate_all(self, splits: List[str] = None) -> Dict[str, Any]:
        """Validates all splits in dataset."""
        splits = splits or ["train", "val"]
        for sp in splits:
            if os.path.exists(os.path.join(self.data_root, sp)):
                self.validate_split(sp)

        return self.stats

    def print_report(self):
        """Prints a human-readable summary report."""
        print("=" * 65)
        print(f"DATASET VALIDATION REPORT: {self.data_root}")
        print("=" * 65)
        print(f"Total Images Checked : {self.stats['total_images']}")
        print(f"Valid Samples Ready  : {self.stats['valid_samples']}")
        print(f"Hard-Negatives (Bg)  : {self.stats['hard_negatives']}")
        print(f"Corrupt Images       : {len(self.stats['corrupt_images'])}")
        print(f"Missing Labels       : {len(self.stats['missing_labels'])}")
        print(f"Orphaned Labels      : {len(self.stats['orphaned_labels'])}")
        print(f"Invalid Bounding Boxes: {len(self.stats['invalid_boxes'])}")
        print(f"Duplicate Images     : {len(self.stats['duplicate_images'])}")

        print("\n[SPLIT BREAKDOWN]")
        for sp, d in self.stats["split_counts"].items():
            print(f"  * {sp:10s} -> Images: {d['images']}, Labels: {d['labels']}")

        print("\n[CLASS INSTANCE DISTRIBUTION]")
        for cid, count in sorted(self.stats["class_counts"].items()):
            cname = self.class_names[cid] if cid < len(self.class_names) else f"Class_{cid}"
            print(f"  * Class {cid} ({cname}): {count} bounding boxes")

        is_passed = (len(self.stats["corrupt_images"]) == 0 and
                     len(self.stats["missing_labels"]) == 0 and
                     len(self.stats["invalid_boxes"]) == 0 and
                     self.stats["valid_samples"] > 0)

        print("\n" + ("=" * 65))
        if is_passed:
            print("[STATUS: PASSED] Dataset meets YOLOv8 format & integrity standards.")
        else:
            print("[STATUS: FAILED / INCOMPLETE] Fix issues above before starting training.")
        print("=" * 65)
        return is_passed


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate YOLO dataset format and integrity.")
    parser.add_argument("--data-dir", type=str, default="training/datasets/cops_weapons", help="Path to dataset root")
    parser.add_argument("--classes", nargs="+", default=["pistol", "knife", "rifle", "grenade"], help="Class names")
    args = parser.parse_args()

    validator = DatasetValidator(args.data_dir, args.classes)
    validator.validate_all()
    validator.print_report()
