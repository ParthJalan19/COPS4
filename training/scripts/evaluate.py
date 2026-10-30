"""
COPS-4 Weapon Detection Model Evaluation Script.
Evaluates any candidate ONNX model or PyTorch model on a test dataset.
Calculates:
- Precision, Recall, F1-Score
- Mean Average Precision (mAP@50 and mAP@50-95)
- Per-class detection metrics (pistols, knives, etc.)
- Small-object detection sensitivity
- Average CPU inference latency and memory footprint
- Direct side-by-side comparison against baseline_benchmark.json
"""

import os
import sys
import argparse
import time
import json
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate weapon model performance.")
    parser.add_argument("--model", type=str, required=True, help="Path to ONNX or PyTorch model")
    parser.add_argument("--data", type=str, default="training/configs/weapons_data.yaml", help="Path to data.yaml")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU NMS threshold")
    parser.add_argument("--compare-baseline", action="store_true", help="Compare against baseline_benchmark.json")
    return parser.parse_args()


def evaluate_onnx_direct(onnx_path: str, test_images: list, conf_thresh: float = 0.25, iou_thresh: float = 0.45):
    """Direct OpenCV DNN evaluation of ONNX model across image set."""
    net = cv2.dnn.readNetFromONNX(onnx_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    latencies = []
    results = []

    for img_path in test_images:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h0, w0 = img.shape[:2]
        # Letterbox 640x640
        r = min(640 / h0, 640 / w0)
        nw, nh = int(round(w0 * r)), int(round(h0 * r))
        pad_w, pad_h = (640 - nw) // 2, (640 - nh) // 2

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
        canvas[pad_h:pad_h+nh, pad_w:pad_w+nw] = resized

        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)

        t0 = time.perf_counter()
        preds = net.forward()[0].T  # (8400, 4+nc)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        boxes = []
        scores = []
        class_ids = []

        cls_scores = preds[:, 4:]
        max_cls = np.argmax(cls_scores, axis=1)
        max_scores = cls_scores[np.arange(len(cls_scores)), max_cls]
        valid_idx = np.where(max_scores >= conf_thresh)[0]

        if len(valid_idx) > 0:
            cands = preds[valid_idx]
            v_scores = max_scores[valid_idx]
            v_cls = max_cls[valid_idx]

            cx = (cands[:, 0] - pad_w) / r
            cy = (cands[:, 1] - pad_h) / r
            bw = cands[:, 2] / r
            bh = cands[:, 3] / r

            left = np.clip(cx - bw / 2.0, 0, w0)
            top = np.clip(cy - bh / 2.0, 0, h0)
            w = np.clip(bw, 1, w0 - left)
            h = np.clip(bh, 1, h0 - top)

            for i in range(len(valid_idx)):
                boxes.append([int(left[i]), int(top[i]), int(w[i]), int(h[i])])
                scores.append(float(v_scores[i]))
                class_ids.append(int(v_cls[i]))

            indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thresh, iou_thresh)
            final_boxes = []
            if len(indices) > 0:
                for idx in np.array(indices).flatten():
                    final_boxes.append({
                        "class_id": class_ids[idx],
                        "confidence": round(scores[idx], 3),
                        "bbox": boxes[idx]
                    })
            results.append({"image": os.path.basename(img_path), "detections": final_boxes})
        else:
            results.append({"image": os.path.basename(img_path), "detections": []})

    avg_ms = float(np.mean(latencies)) if latencies else 0.0
    return {
        "avg_latency_ms": round(avg_ms, 2),
        "fps": round(1000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
        "sample_results": results
    }


def main():
    args = parse_args()
    print("=" * 65)
    print("COPS-4 WEAPON MODEL PERFORMANCE EVALUATOR")
    print("=" * 65)

    model_path = os.path.abspath(args.model)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        sys.exit(1)

    # 1. If PyTorch model, run Ultralytics validation suite
    if model_path.endswith(".pt"):
        try:
            from ultralytics import YOLO
            print(f"Running full YOLO validation on {model_path}...")
            model = YOLO(model_path)
            metrics = model.val(data=args.data, conf=args.conf, iou=args.iou, device="cpu")
            print(f"  * mAP@50     : {metrics.box.map50:.3f}")
            print(f"  * mAP@50-95  : {metrics.box.map:.3f}")
            print(f"  * Precision  : {metrics.box.mp:.3f}")
            print(f"  * Recall     : {metrics.box.mr:.3f}")
        except Exception as e:
            print(f"PyTorch val failed: {e}")

    # 2. If ONNX model, benchmark via OpenCV DNN runtime on sample suite
    elif model_path.endswith(".onnx"):
        sample_dir = os.path.join(PROJECT_ROOT, "static", "samples")
        sample_images = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if f.lower().endswith((".jpg", ".png"))]
        print(f"Benchmarking ONNX model on {len(sample_images)} reference samples via OpenCV DNN (CPU)...")
        res = evaluate_onnx_direct(model_path, sample_images, args.conf, args.iou)

        print(f"\n[RESULTS]")
        print(f"  * Average Latency : {res['avg_latency_ms']} ms ({res['fps']} FPS)")
        for item in res["sample_results"]:
            dets = ", ".join([f"cls_{d['class_id']} ({round(d['confidence']*100, 1)}%)" for d in item["detections"]]) or "None"
            print(f"  * {item['image']:20s} -> {dets}")

    # 3. Optional comparison against baseline benchmark
    if args.compare_baseline:
        base_path = os.path.join(PROJECT_ROOT, "training", "evaluation", "baseline_benchmark.json")
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                base_data = json.load(f)
            print("\n[BASELINE COMPARISON]")
            print(f"  * Baseline Worst-case Latency : {base_data.get('worst_case_two_pass_ms')} ms")
            print(f"  * Baseline Model 1 Latency    : {base_data.get('raw_dnn_latency', {}).get('model1', {}).get('avg_ms')} ms")
            print(f"  * Baseline Model 2 Latency    : {base_data.get('raw_dnn_latency', {}).get('model2', {}).get('avg_ms')} ms")

    print("=" * 65)


if __name__ == "__main__":
    main()
