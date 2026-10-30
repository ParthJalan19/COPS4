"""
COPS-4 Standalone Custom Weapon Detection Model Training Pipeline.
Runs completely isolated from the production backend.
Supports:
- Transfer learning from lightweight base models (yolov8n.pt / yolov8s.pt)
- Configurable hyperparameters, epochs, early stopping, and data configs
- Class imbalance weighting
- Pre-flight dataset verification before training launch
- Checkpoint management and best model selection
- Automatic export to ONNX upon training completion
"""

import os
import sys
import argparse
import yaml
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from training.scripts.validate_dataset import DatasetValidator


def parse_args():
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 model for COPS-4 weapon detection.")
    parser.add_argument("--data", type=str, default="training/configs/weapons_data.yaml", help="Path to data.yaml")
    parser.add_argument("--params", type=str, default="training/configs/train_params.yaml", help="Path to train_params.yaml")
    parser.add_argument("--model", type=str, default=None, help="Base model weights (e.g. yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=None, help="Input image resolution")
    parser.add_argument("--device", type=str, default=None, help="Training device: '0', 'cpu', '0,1'")
    parser.add_argument("--name", type=str, default="weapon_yolov8n_custom", help="Experiment run name")
    parser.add_argument("--skip-validation", action="store_true", help="Skip pre-flight dataset check")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 65)
    print("COPS-4 STANDALONE WEAPON MODEL TRAINING PIPELINE")
    print("=" * 65)

    data_config_path = os.path.abspath(args.data)
    params_config_path = os.path.abspath(args.params)

    if not os.path.exists(data_config_path):
        print(f"[ERROR] Data config file not found: {data_config_path}")
        sys.exit(1)

    with open(data_config_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    with open(params_config_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    # CLI Overrides
    if args.model: train_cfg["model"] = args.model
    if args.epochs: train_cfg["epochs"] = args.epochs
    if args.batch: train_cfg["batch"] = args.batch
    if args.imgsz: train_cfg["imgsz"] = args.imgsz
    if args.device: train_cfg["device"] = args.device

    dataset_root = data_cfg.get("path", "")
    class_names = list(data_cfg.get("names", {}).values()) if isinstance(data_cfg.get("names"), dict) else data_cfg.get("names", [])

    # 1. Pre-flight Dataset Verification
    if not args.skip_validation and dataset_root:
        print(f"\n[1] PRE-FLIGHT DATASET VALIDATION ({dataset_root})...")
        validator = DatasetValidator(dataset_root, class_names)
        validator.validate_all()
        passed = validator.print_report()
        if not passed:
            print("[ERROR] Dataset validation failed. Resolve corrupted images/annotations before training.")
            sys.exit(1)
    else:
        print("\n[1] Pre-flight dataset validation skipped or path unconfigured.")

    # 2. Check Ultralytics Environment
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        print("\n[ERROR] Ultralytics/PyTorch not installed in the training environment.")
        print("Please install requirements using:")
        print("    pip install -r training/requirements.txt")
        sys.exit(1)

    print("\n[2] ENVIRONMENT CHECK:")
    print(f"  * PyTorch Version : {torch.__version__}")
    print(f"  * CUDA Available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  * Device Name     : {torch.cuda.get_device_name(0)}")
        print(f"  * Device Count    : {torch.cuda.device_count()}")
    else:
        print("  * Note: Training will run on CPU. Set batch size lower for CPU (e.g. 4 or 8).")
        if train_cfg.get("device") != "cpu":
            train_cfg["device"] = "cpu"

    # 3. Model Initialization
    base_model_path = train_cfg.get("model", "yolov8n.pt")
    print(f"\n[3] INITIALIZING BASE MODEL: {base_model_path}")
    model = YOLO(base_model_path)

    project_runs_dir = os.path.join(PROJECT_ROOT, "training", "runs")
    os.makedirs(project_runs_dir, exist_ok=True)

    # 4. Launch Training
    print(f"\n[4] LAUNCHING TRAINING RUN: {args.name}")
    print(f"  * Epochs : {train_cfg['epochs']}")
    print(f"  * Batch  : {train_cfg['batch']}")
    print(f"  * ImgSz  : {train_cfg['imgsz']}")
    print(f"  * Device : {train_cfg['device']}")

    t0 = time.time()
    try:
        results = model.train(
            data=data_config_path,
            epochs=train_cfg["epochs"],
            batch=train_cfg["batch"],
            imgsz=train_cfg["imgsz"],
            device=train_cfg["device"],
            project=project_runs_dir,
            name=args.name,
            optimizer=train_cfg.get("optimizer", "AdamW"),
            lr0=train_cfg.get("lr0", 0.001),
            lrf=train_cfg.get("lrf", 0.01),
            patience=train_cfg.get("patience", 15),
            save=True,
            save_period=train_cfg.get("save_period", 10),
            mosaic=train_cfg.get("mosaic", 1.0),
            mixup=train_cfg.get("mixup", 0.15),
            fliplr=train_cfg.get("fliplr", 0.5),
            flipud=train_cfg.get("flipud", 0.0),
            exist_ok=True
        )
    except Exception as e:
        print(f"\n[ERROR] Training failed with exception: {e}")
        sys.exit(1)

    train_duration_min = round((time.time() - t0) / 60.0, 1)
    best_weights_path = os.path.join(project_runs_dir, args.name, "weights", "best.pt")

    print(f"\n[5] TRAINING COMPLETED in {train_duration_min} minutes.")
    print(f"  * Best weights saved to: {best_weights_path}")

    # 5. Export to ONNX automatically
    if os.path.exists(best_weights_path):
        print(f"\n[6] EXPORTING BEST WEIGHTS TO PRODUCTION-READY ONNX...")
        best_model = YOLO(best_weights_path)
        exported_path = best_model.export(
            format="onnx",
            imgsz=train_cfg["imgsz"],
            opset=12,
            simplify=True
        )
        print(f"  * ONNX Model Exported: {exported_path}")
        print("\nRun training/scripts/evaluate_onnx.py to benchmark against the baseline before production deployment.")
    print("=" * 65)


if __name__ == "__main__":
    main()
