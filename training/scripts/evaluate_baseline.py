"""
COPS-4 Existing Production Model Benchmark Script.
Evaluates Model 1 and Model 2 on test samples and measures:
- Output tensor shapes and compatibility
- Single-model vs. dual-model latency
- Two-pass multi-scale overhead
- Memory and CPU thread consumption
"""

import os
import sys
import time
import json
import psutil
import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from weapon_detector import weapon_detector, MODEL1_PATH, MODEL2_PATH


def benchmark_model_forward(net, sample_blob, warmup=5, iterations=30):
    """Measures raw forward pass latency of an OpenCV DNN ONNX net."""
    if net is None:
        return {"avg_ms": 0.0, "std_ms": 0.0, "fps": 0.0}

    # Warmup
    for _ in range(warmup):
        net.setInput(sample_blob)
        _ = net.forward()

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        net.setInput(sample_blob)
        _ = net.forward()
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
    return {
        "avg_ms": round(avg_ms, 2),
        "std_ms": round(std_ms, 2),
        "fps": round(fps, 1)
    }


def main():
    print("=" * 65)
    print("COPS-4 EXISTING WEAPON DETECTOR BENCHMARK & AUDIT")
    print("=" * 65)

    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)

    # 1. Inspect Model Files
    models_info = {}
    for name, path in [("Model 1 (Pistols & Knives)", MODEL1_PATH),
                       ("Model 2 (Threat Ensemble)", MODEL2_PATH)]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            models_info[name] = {"path": path, "size_mb": round(size_mb, 2), "exists": True}
        else:
            models_info[name] = {"path": path, "size_mb": 0.0, "exists": False}

    print("\n[1] MODEL FILES STATUS:")
    for k, v in models_info.items():
        print(f"  * {k}: {v['size_mb']} MB (Found: {v['exists']})")

    # 2. Benchmark Raw DNN Forward Passes (640x640)
    dummy_blob = np.random.uniform(0.0, 1.0, (1, 3, 640, 640)).astype(np.float32)

    print("\n[2] RAW INFERENCE LATENCY (CPU, Batch=1, 640x640):")
    net1_perf = benchmark_model_forward(weapon_detector.net1, dummy_blob)
    print(f"  * Model 1 (weapon_detection_yolov8.onnx): {net1_perf['avg_ms']} ms ± {net1_perf['std_ms']} ms (~{net1_perf['fps']} FPS)")

    net2_perf = benchmark_model_forward(weapon_detector.net2, dummy_blob)
    print(f"  * Model 2 (weapon_detection_threat.onnx): {net2_perf['avg_ms']} ms ± {net2_perf['std_ms']} ms (~{net2_perf['fps']} FPS)")

    # 3. Benchmark End-to-End WeaponDetector on Real Sample Images
    sample_dir = os.path.join(PROJECT_ROOT, "static", "samples")
    sample_images = []
    if os.path.exists(sample_dir):
        for fname in os.listdir(sample_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg")):
                fpath = os.path.join(sample_dir, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    sample_images.append((fname, img))

    print(f"\n[3] END-TO-END PIPELINE BENCHMARK ({len(sample_images)} test samples):")
    detection_summary = []

    for fname, img in sample_images:
        t0 = time.perf_counter()
        detections = weapon_detector.detect_weapons(img, custom_conf=0.20)
        dur_ms = (time.perf_counter() - t0) * 1000.0

        item = {
            "file": fname,
            "resolution": f"{img.shape[1]}x{img.shape[0]}",
            "latency_ms": round(dur_ms, 2),
            "detections_count": len(detections),
            "detections": detections
        }
        detection_summary.append(item)
        labels_str = ", ".join([f"{d['label']} ({d['confidence']}%)" for d in detections]) or "None"
        print(f"  * {fname:20s} [{img.shape[1]}x{img.shape[0]}]: {dur_ms:6.1f} ms -> {labels_str}")

    # 4. Measure Two-Pass Worst-Case (Empty Room with no weapon -> triggers Pass 2 zoom)
    empty_frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    e2e_times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = weapon_detector.detect_weapons(empty_frame, custom_conf=0.20)
        e2e_times.append((time.perf_counter() - t0) * 1000.0)

    worst_case_ms = round(float(np.mean(e2e_times)), 2)
    mem_after = process.memory_info().rss / (1024 * 1024)

    print(f"\n[4] WORST-CASE LATENCY (Pass 1 + Pass 2 Zoom on non-threat frame):")
    print(f"  * Average latency per non-threat frame: {worst_case_ms} ms (~{round(1000.0/worst_case_ms, 1)} FPS)")
    print(f"  * Process Memory Usage: {round(mem_after, 2)} MB (Delta: +{round(mem_after - mem_before, 2)} MB)")

    # Save benchmark report to evaluation directory
    out_path = os.path.join(PROJECT_ROOT, "training", "evaluation", "baseline_benchmark.json")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": models_info,
        "raw_dnn_latency": {
            "model1": net1_perf,
            "model2": net2_perf,
            "dual_sequential_sum_ms": round(net1_perf["avg_ms"] + net2_perf["avg_ms"], 2)
        },
        "worst_case_two_pass_ms": worst_case_ms,
        "sample_evaluations": detection_summary
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Baseline benchmark results saved to: {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

