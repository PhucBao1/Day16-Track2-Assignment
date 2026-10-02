"""Lab 16 - LightGBM benchmark on Credit Card Fraud Detection (CPU node).

Usage:
    python3 benchmark.py [--data ~/ml-benchmark/creditcard.csv] [--out benchmark_result.json]
"""
import argparse
import json
import os
import platform
import time
import warnings
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEED = 42

# lightgbm>=4.7 deprecates eval_set in favour of eval_X/eval_y; keep eval_set for older versions
warnings.filterwarnings("ignore", message=".*'eval_set' is deprecated.*")


def parse_args():
    parser = argparse.ArgumentParser(description="LightGBM fraud detection benchmark")
    parser.add_argument("--data", default=os.path.expanduser("~/ml-benchmark/creditcard.csv"))
    parser.add_argument("--out", default="benchmark_result.json")
    parser.add_argument("--latency-runs", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load data
    t0 = time.perf_counter()
    df = pd.read_csv(args.data)
    load_time = time.perf_counter() - t0

    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Train / test 80/20, then carve a validation set out of train for early stopping
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=SEED
    )

    # 2. Train
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=31,
        # Fraud is ~0.17% of rows: with the default min_child_weight (1e-3) leaves holding a handful
        # of positives get extreme values and validation AUC peaks at iteration 1. Larger leaves fix it.
        min_child_samples=50,
        min_child_weight=1.0,
        metric="auc",  # early-stop on AUC only (logloss alone stops too early on rare-class data)
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    t0 = time.perf_counter()
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    train_time = time.perf_counter() - t0

    # 3. Evaluate
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc_roc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "f1_score": f1_score(y_test, pred, zero_division=0),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
    }

    # 4. Inference latency (1 row) and throughput (1000 rows)
    one_row = X_test.iloc[[0]]
    model.predict_proba(one_row)  # warm-up
    latencies = []
    for _ in range(args.latency_runs):
        t0 = time.perf_counter()
        model.predict_proba(one_row)
        latencies.append(time.perf_counter() - t0)
    latency_ms = float(np.mean(latencies) * 1000)
    latency_p95_ms = float(np.percentile(latencies, 95) * 1000)

    batch = X_test.iloc[:1000]
    t0 = time.perf_counter()
    model.predict_proba(batch)
    batch_time = time.perf_counter() - t0

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "cpu_count": os.cpu_count(),
        },
        "dataset": {
            "path": args.data,
            "rows": int(len(df)),
            "features": int(X.shape[1]),
            "fraud_ratio": float(y.mean()),
            "train_rows": int(len(X_tr)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
        },
        "load_time_s": load_time,
        "train_time_s": train_time,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "metrics": metrics,
        "inference": {
            "latency_1_row_ms": latency_ms,
            "latency_1_row_p95_ms": latency_p95_ms,
            "batch_1000_rows_ms": batch_time * 1000,
            "throughput_rows_per_s": len(batch) / batch_time,
        },
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    rows = [
        ("Thoi gian load data", f"{load_time:.3f} s"),
        ("Thoi gian training", f"{train_time:.3f} s"),
        ("Best iteration", str(result["best_iteration"])),
        ("AUC-ROC", f"{metrics['auc_roc']:.4f}"),
        ("Accuracy", f"{metrics['accuracy']:.4f}"),
        ("F1-Score", f"{metrics['f1_score']:.4f}"),
        ("Precision", f"{metrics['precision']:.4f}"),
        ("Recall", f"{metrics['recall']:.4f}"),
        ("Inference latency (1 row)", f"{latency_ms:.3f} ms (p95 {latency_p95_ms:.3f} ms)"),
        ("Inference throughput (1000 rows)",
         f"{batch_time * 1000:.2f} ms ({len(batch) / batch_time:,.0f} rows/s)"),
    ]
    width = max(len(name) for name, _ in rows)
    print("=" * 60)
    print(f"LightGBM benchmark - {platform.node()} ({os.cpu_count()} vCPU)")
    print("=" * 60)
    for name, value in rows:
        print(f"{name:<{width}} : {value}")
    print("=" * 60)
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
