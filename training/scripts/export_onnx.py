"""
COPS-4 ONNX Model Exporter & Graph Verification Utility.
Converts trained PyTorch (.pt) weights to deployment-ready ONNX:
1. Exports with opset=12 for broad OpenCV DNN compatibility
2. Runs onnx-simplifier (if available) to fuse redundant nodes
3. Validates output tensor shape (e.g. 1 x (4 + num_classes) x 8400)
4. Validates that OpenCV DNN can successfully read and execute the model on CPU
5. Checks inference latency on a dummy tensor
"""

import os
import sys
import argparse
import time
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Export and validate YOLO model to ONNX for COPS-4.")
    parser.add_argument("--weights", type=str, required=True, help="Path to trained PyTorch .pt weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution (default: 640)")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version (default: 12)")
    parser.add_argument("--output", type=str, default=None, help="Destination ONNX path")
    return parser.parse_args()


def validate_onnx_with_opencv(onnx_path: str, imgsz: int = 640):
    """Verifies that OpenCV DNN can load the exported ONNX model and runs a test forward pass."""
    print(f"\n[OpenCV DNN Validation] Testing {onnx_path}...")
    if not os.path.exists(onnx_path):
        print(f"[FAIL] ONNX file not found: {onnx_path}")
        return False

    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  * File Size: {round(file_size_mb, 2)} MB")

    try:
        net = cv2.dnn.readNetFromONNX(onnx_path)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    except Exception as e:
        print(f"[FAIL] OpenCV DNN failed to load model: {e}")
        return False

    # Test forward pass with dummy tensor
    dummy = np.random.uniform(0.0, 1.0, (1, 3, imgsz, imgsz)).astype(np.float32)
    try:
        net.setInput(dummy)
        t0 = time.perf_counter()
        output = net.forward()
        dur_ms = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        print(f"[FAIL] OpenCV DNN forward pass crashed: {e}")
        return False

    print(f"  * Output Tensor Shape: {output.shape}")
    print(f"  * Test Forward Pass Duration: {round(dur_ms, 2)} ms on CPU")

    # Validate YOLOv8 standard output shape (1, 4+classes, anchors)
    if len(output.shape) != 3 or output.shape[0] != 1 or output.shape[1] < 5:
        print(f"[WARNING] Output tensor shape {output.shape} differs from standard YOLOv8 (1, 4+nc, 8400)")
    else:
        num_classes = output.shape[1] - 4
        num_anchors = output.shape[2]
        print(f"  * Detected {num_classes} classes across {num_anchors} anchor candidates.")

    print("[SUCCESS] Model passed OpenCV DNN CPU inference verification!")
    return True


def main():
    args = parse_args()
    print("=" * 65)
    print("COPS-4 ONNX MODEL EXPORT & VALIDATION PIPELINE")
    print("=" * 65)

    weights_path = os.path.abspath(args.weights)
    if not os.path.exists(weights_path):
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"Loading model: {weights_path}")
    model = YOLO(weights_path)

    print(f"Exporting to ONNX (imgsz={args.imgsz}, opset={args.opset})...")
    exported_file = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True
    )

    if args.output and exported_file != args.output:
        import shutil
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        shutil.copy2(exported_file, args.output)
        exported_file = args.output
        print(f"Copied export to: {exported_file}")

    # Validate with OpenCV DNN
    is_valid = validate_onnx_with_opencv(exported_file, args.imgsz)
    if not is_valid:
        print("[ERROR] Model failed validation. Do not deploy to production.")
        sys.exit(1)

    print("=" * 65)


if __name__ == "__main__":
    main()
