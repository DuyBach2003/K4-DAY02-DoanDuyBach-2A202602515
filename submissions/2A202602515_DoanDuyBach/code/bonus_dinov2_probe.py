"""Điểm thưởng (RUBRIC mục 2): linear probe trên đặc trưng ĐÓNG BĂNG của mô hình nền tảng DINOv2 ViT-S/14 (tự giám sát,
LVD-142M), so với DeiT-S tiền huấn luyện có giám sát trên ImageNet-1k (cùng cỡ, cùng giao thức probe) và với các mạng
tinh chỉnh toàn bộ ở Bước 1. Chỉ dùng train (fit) và val (đánh giá) của fold 0; không đụng test.

Giao thức: ảnh qua transform val (Resize 256 + CenterCrop 224, không augmentation); đặc trưng = [token CLS, trung bình
token patch] của lớp cuối (như cách đánh giá linear của DINOv2); chuẩn hoá theo train; hồi quy logistic đa lớp, C chọn
bằng 5-fold CV phân tầng trên TRAIN theo macro-F1 (không nhìn val).

    python code/bonus_dinov2_probe.py   (từ thư mục bài nộp) -> logs/bonus_dinov2_probe.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402
import dataset as D  # noqa: E402
from eval import compute_metrics  # noqa: E402
from step3_inference import loader_for  # noqa: E402

MODELS = {"dinov2_vits14": "vit_small_patch14_dinov2.lvd142m", "deit_s_in1k": "deit_small_patch16_224.fb_in1k"}
CS = [0.01, 0.1, 1.0]


def features(model, loader, device):
    import torch

    feats, ys = [], []
    with torch.inference_mode():
        for x, y, _ in loader:
            t = model.forward_features(x.to(device))  # (B, token tiền tố + patch, C), đã qua norm cuối
            f = torch.cat([t[:, 0], t[:, model.num_prefix_tokens:].mean(1)], 1)
            feats.append(f.float().cpu())
            ys.append(y)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()


def main():
    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    device = TR.get_device()
    tr, va, _ = D.load_split("../../data/labels", 0)
    out = {"protocol": "transform val 224, đặc trưng [CLS, mean patch] lớp cuối, StandardScaler + LogisticRegression; "
                       "C chọn bằng 5-fold CV trên train (macro-F1)", "Cs": CS, "results": {}}
    for key, name in MODELS.items():
        m = timm.create_model(name, pretrained=True, num_classes=0, dynamic_img_size=True).to(device).eval()
        t0 = time.perf_counter()
        xtr, ytr = features(m, loader_for(tr, "../../data/images", 224, True), device)
        xva, yva = features(m, loader_for(va, "../../data/images", 224, True), device)
        t_feat = time.perf_counter() - t0
        gs = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)),
                          {"logisticregression__C": CS}, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                          scoring="f1_macro", n_jobs=2)
        gs.fit(xtr, ytr)
        p = gs.predict_proba(xva)
        r = compute_metrics(yva, p.argmax(1), p)
        out["results"][key] = {
            "timm": name, "weight_tag": m.pretrained_cfg.get("tag", ""), "feat_dim": int(xtr.shape[1]),
            "params_m": sum(q.numel() for q in m.parameters()) / 1e6, "best_C": gs.best_params_["logisticregression__C"],
            "cv_macro_f1_train": float(gs.best_score_), "val_macro_f1": r["macro_f1"], "val_top1": r["top1"],
            "val_ece": r["ece"], "val_f1_chinee": float(r["f1"][0]), "val_f1_snake": float(r["f1"][7]),
            "feature_time_s": t_feat, "n_train": len(ytr), "n_val": len(yva)}
        print(key, json.dumps(out["results"][key]), flush=True)
        del m
    Path("logs/bonus_dinov2_probe.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
