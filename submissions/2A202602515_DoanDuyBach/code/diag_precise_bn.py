"""Chẩn đoán giả thuyết lệch thống kê BatchNorm (Bước 1): đánh giá lại checkpoint của các CNN có BN trên VAL
trước/sau khi tính lại thống kê BN bằng toàn bộ ảnh TRAIN qua transform val (precise-BN, không dùng nhãn).

    python code/diag_precise_bn.py      (từ thư mục bài nộp) -> logs/diag_precise_bn.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402
import dataset as D  # noqa: E402
from eval import compute_metrics  # noqa: E402
from inference import predict_views, recalibrate_bn  # noqa: E402
from step3_inference import load_model, loader_for  # noqa: E402

RUNS = {"B01": "resnet50", "B05": "efficientnet_b0", "B06": "mobilenetv3_large_100"}


def score(model, loader, device):
    _, y, (z,) = predict_views(model, loader, device)
    p = TR.softmax_np(z)
    m = compute_metrics(y, p.argmax(1), p)
    return {"macro_f1": m["macro_f1"], "top1": m["top1"], "ece": m["ece"]}


def main():
    device = TR.get_device()
    tr, va, _ = D.load_split("../../data/labels", 0)
    val_loader = loader_for(va, "../../data/images", 224, True)
    cal_loader = loader_for(tr, "../../data/images", 224, True)
    out = {}
    for exp, bb in RUNS.items():
        m = load_model(exp, bb, 0, device)
        before = score(m, val_loader, device)
        pbn = recalibrate_bn(m, cal_loader, device)
        after = score(pbn, val_loader, device)
        out[exp] = {"backbone": bb, "before": before, "after_precise_bn": after, "n_bn": pbn.n_recalibrated_bn,
                    "calibration_images": len(tr)}
        print(exp, json.dumps(out[exp]), flush=True)
    Path("logs/diag_precise_bn.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
