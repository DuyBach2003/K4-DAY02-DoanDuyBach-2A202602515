"""Phân tích sau chung kết, CHỈ trên VAL: tách đóng góp của công thức huấn luyện và của phương pháp suy luận.

Thiết kế 2×2, 3 seed mỗi ô: {T00 (công thức nền), F01 (label smoothing + crop nhẹ)} × {I00: Resize256 + CenterCrop224,
I04_288: ảnh đầy đủ resize 288}, cùng checkpoint của Bước 4. Không đọc ảnh test, không đổi cấu hình chung kết (đã chốt
và đã chạy test trước khi phân tích này được chạy). Macro-F1 không phụ thuộc temperature (T không đổi argmax).

    python code/diag_factorial_val.py      (từ thư mục bài nộp) -> logs/diag_factorial_val.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402
import dataset as D  # noqa: E402
from eval import compute_metrics  # noqa: E402
from inference import predict_views  # noqa: E402
from step3_inference import load_model, loader_for  # noqa: E402

BACKBONE = "deit_small_patch16_224"
EXPS = ("T00", "F01")
SEEDS = (0, 1, 2)
INFER = {"I00_crop224": (224, True), "I04_full288": (288, False)}


def mean_std(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "values": [float(x) for x in v]}


def main():
    device = TR.get_device()
    _, va, _ = D.load_split("../../data/labels", 0)
    loaders = {k: loader_for(va, "../../data/images", size, crop) for k, (size, crop) in INFER.items()}
    f1 = {}
    out = {"split": "val", "n_images": len(va), "cells": {}}
    for exp in EXPS:
        for seed in SEEDS:
            m = load_model(exp, BACKBONE, seed, device)
            for k, loader in loaders.items():
                _, y, (z,) = predict_views(m, loader, device)
                p = TR.softmax_np(z)
                r = compute_metrics(y, p.argmax(1), p)
                rec = {"macro_f1": r["macro_f1"], "top1": r["top1"], "f1_chinee": float(r["f1"][0]),
                       "f1_snake": float(r["f1"][7])}
                out["cells"].setdefault(f"{exp}|{k}", {})[seed] = rec
                f1[(exp, k, seed)] = r["macro_f1"]
                print(exp, seed, k, json.dumps(rec), flush=True)
    cell = {f"{e}|{k}": mean_std([f1[(e, k, s)] for s in SEEDS]) for e in EXPS for k in INFER}
    a, b = INFER
    out["summary_macro_f1"] = cell
    # suy luận: so sánh cặp trên cùng checkpoint (288 - 224); công thức: hiệu hai trung bình, s = std lớn hơn
    out["effect_inference"] = {e: mean_std([f1[(e, b, s)] - f1[(e, a, s)] for s in SEEDS]) for e in EXPS}
    out["effect_recipe"] = {k: {"delta": cell[f"F01|{k}"]["mean"] - cell[f"T00|{k}"]["mean"],
                                "s": max(cell[f"F01|{k}"]["std"], cell[f"T00|{k}"]["std"])} for k in INFER}
    out["interaction"] = out["effect_recipe"][b]["delta"] - out["effect_recipe"][a]["delta"]
    Path("logs/diag_factorial_val.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("summary_macro_f1", "effect_inference", "effect_recipe", "interaction")},
                     indent=1))


if __name__ == "__main__":
    main()
