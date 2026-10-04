"""Bước 3: so sánh phương pháp suy luận trên VAL (không huấn luyện lại) + đo độ trễ.

Chạy từ thư mục bài nộp:
    python code/step3_inference.py --exp T00 --backbone efficientnet_b0 [--ema-exp T10] [--ensemble B01 B02 ...]
Kết quả: logs/step3_inference.json, logs/step3_latency.json, figures/step3_tradeoff.png

Phương pháp (mã theo GUIDE mục 4):
  I00 1 view (Resize 256 + CenterCrop 224)       I01 TTA lật ngang (K=2)
  I02a 5-crop 224 từ ảnh 256 (K=5)               I02b 5-crop + lật (K=10)
  I02c đa tỉ lệ 224/256/288 trên ảnh đầy đủ (K=3)
  I03 gộp logit (thay cho gộp xác suất) cho I01/I02b/I02c
  I04 dò độ phân giải kiểm tra: ảnh đầy đủ resize về 224/256/288/320 (1 view)
  I05 ensemble các backbone của Bước 1 (trung bình xác suất từ logit val đã lưu)
  I06 trọng số EMA (lấy từ thí nghiệm huấn luyện có EMA, không tốn thêm khi suy luận)
  I07 temperature scaling: T khớp trên val; ECE trước/sau (thêm ước lượng chéo 2 nửa val để tránh lạc quan)
  I08 gộp BN vào conv; FP16 (model.half()) và AMP
  I09 precise-BN: tính lại thống kê BatchNorm bằng ảnh train qua transform val (không tốn thêm lúc suy luận)
Quy ước gộp mặc định: trung bình xác suất ("prob"); I03 so với trung bình logit.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402  (đặt sys.path tới eval.py)
from eval import compute_metrics  # noqa: E402


def metrics_of(probs, y):
    m = compute_metrics(y, probs.argmax(1), probs)
    return {"macro_f1": m["macro_f1"], "top1": m["top1"], "ece": m["ece"], "nll": m["nll"],
            "bal_acc": m["balanced_acc"], "f1_chinee": float(m["f1"][0]), "f1_snake": float(m["f1"][7])}


def load_model(exp: str, backbone: str, seed: int, device):
    import torch

    import timm

    # ViT/DeiT: dynamic_img_size=True nội suy pos-embed để chạy được ở độ phân giải khác 224 (I02c, I04);
    # ở đúng 224 đầu ra không đổi. CNN chạy được mọi kích thước nhờ global pooling.
    kw = {"dynamic_img_size": True} if any(k in backbone for k in ("vit", "deit")) else {}
    m = timm.create_model(backbone, pretrained=False, num_classes=9, **kw)
    m.load_state_dict(torch.load(Path("runs") / exp / f"seed{seed}" / "best.pt", map_location="cpu"))
    return m.to(device).eval()


def loader_for(df, images_dir, size, crop: bool):
    """crop=True: Resize(size/0.875)+CenterCrop(size) (chuẩn val). crop=False: ảnh đầy đủ resize về size."""
    from torchvision import transforms as T

    import dataset as D

    if crop:
        tf = D.build_transforms(False, size)
    else:
        tf = T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)])
    return D.make_loader(df, images_dir, tf, 64, False, None, NUM_WORKERS)


NUM_WORKERS = 4


def cross_fit_ece(logits, y, seed=0):
    """ECE sau temperature scaling ước lượng chéo: khớp T trên một nửa val, đo ECE trên nửa kia, rồi đảo."""
    from inference import apply_temperature, fit_temperature
    from eval import ece_score

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a, b = idx[: len(y) // 2], idx[len(y) // 2:]
    eces, Ts = [], []
    for fit, ev in ((a, b), (b, a)):
        T = fit_temperature(logits[fit], y[fit])
        Ts.append(T)
        eces.append(ece_score(apply_temperature(logits[ev], T), y[ev]))
    return float(np.mean(eces)), Ts


def main():
    import torch

    import dataset as D
    from benchmark import latency_report
    from inference import (aggregate_views, apply_temperature, ensemble_probs, fit_temperature, fuse_conv_bn,
                           predict_views, views_hflip, views_multicrop, views_multiscale)

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="thí nghiệm có checkpoint dùng cho suy luận")
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ema-exp", default=None, help="thí nghiệm cùng backbone huấn luyện có EMA (I06)")
    ap.add_argument("--ensemble", nargs="*", default=[], help="exp_id có val_logits.npy (seed 0) để ensemble")
    ap.add_argument("--images-dir", default="../../data/images")
    ap.add_argument("--labels-dir", default="../../data/labels")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="chỉ để chạy thử: dùng N ảnh val đầu tiên")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--skip-latency", action="store_true", help="chỉ để chạy thử")
    args = ap.parse_args()
    global NUM_WORKERS
    if args.limit:
        NUM_WORKERS = 0

    device = torch.device(args.device) if args.device else TR.get_device()
    _, val_df, _ = D.load_split(args.labels_dir, 0)
    full_val_names = val_df["Filename"].tolist()
    if args.limit:
        val_df = val_df.head(args.limit)
    model = load_model(args.exp, args.backbone, args.seed, device)
    res, lat = {}, {}

    def add(code, name, probs, y, k=1, **extra):
        res[code] = {"method": name, "k": k, **metrics_of(probs, y), **extra}
        print(f"{code:10s} {name:45s} F1={res[code]['macro_f1']:.4f} top1={res[code]['top1']:.4f} "
              f"ECE={res[code]['ece']:.4f}", flush=True)

    # --- chạy model: 224 crop (I00/I01), ảnh 256 đầy đủ (5-crop, đa tỉ lệ), các độ phân giải ---
    names, y, (z0, zf) = predict_views(model, loader_for(val_df, args.images_dir, 224, True), device, views_hflip)
    assert names == val_df["Filename"].tolist()
    add("I00", "1 view (Resize256+CenterCrop224)", aggregate_views([z0]), y)
    add("I01", "TTA lật ngang, gộp xác suất", aggregate_views([z0, zf], "prob"), y, 2)
    add("I03a", "TTA lật ngang, gộp logit", aggregate_views([z0, zf], "logit"), y, 2)

    _, _, crops = predict_views(model, loader_for(val_df, args.images_dir, 256, False), device,
                                lambda x: views_multicrop(x, 224, flip=True))
    add("I02a", "5-crop 224 từ ảnh 256, gộp xác suất", aggregate_views(crops[:5], "prob"), y, 5)
    add("I02b", "5-crop + lật (K=10), gộp xác suất", aggregate_views(crops, "prob"), y, 10)
    add("I03b", "5-crop + lật (K=10), gộp logit", aggregate_views(crops, "logit"), y, 10)

    _, _, scales = predict_views(model, loader_for(val_df, args.images_dir, 256, False), device,
                                 lambda x: views_multiscale(x, [224, 256, 288]))
    add("I02c", "đa tỉ lệ 224/256/288 (ảnh đầy đủ), gộp xác suất", aggregate_views(scales, "prob"), y, 3)
    add("I03c", "đa tỉ lệ 224/256/288, gộp logit", aggregate_views(scales, "logit"), y, 3)

    res_logits = {}
    for s in (224, 256, 288, 320):
        _, _, (zs,) = predict_views(model, loader_for(val_df, args.images_dir, s, False), device)
        res_logits[s] = zs
        add(f"I04_{s}", f"độ phân giải kiểm tra {s} (ảnh đầy đủ, 1 view)", aggregate_views([zs]), y, 1,
            img_size=s)

    # --- I05 ensemble backbone (logit val của checkpoint tốt nhất mỗi lần chạy Bước 1) ---
    if args.ensemble:
        probs_list = []
        for e in args.ensemble:
            rd = Path("runs") / e / "seed0"
            assert json.loads((rd / "val_names.json").read_text()) == full_val_names
            probs_list.append(aggregate_views([np.load(rd / "val_logits.npy")[: len(names)]]))
        add("I05", "ensemble " + "+".join(args.ensemble), ensemble_probs(probs_list), y, len(args.ensemble),
            members=args.ensemble)

    # --- I06 EMA ---
    if args.ema_exp:
        ema_model = load_model(args.ema_exp, args.backbone, args.seed, device)
        _, _, (ze,) = predict_views(ema_model, loader_for(val_df, args.images_dir, 224, True), device)
        add("I06", f"trọng số EMA ({args.ema_exp}), 1 view", aggregate_views([ze]), y, 1, model=args.ema_exp)
        del ema_model

    # --- I07 temperature scaling trên I00 và I01 ---
    for base, zs in (("I00", [z0]), ("I01", [z0, zf])):
        zbar = np.mean(zs, 0)  # T áp lên logit trung bình (gộp logit) để có một T duy nhất
        T = fit_temperature(zbar, y)
        ece_cv, Ts = cross_fit_ece(zbar, y)
        add(f"I07_{base}", f"{base} + temperature scaling (T={T:.3f}, khớp trên val)",
            apply_temperature(zbar, T), y, len(zs), T=T, ece_before=metrics_of(aggregate_views([zbar]), y)["ece"],
            ece_after_crossfit=ece_cv, T_halves=Ts)

    # --- I08 gộp BN + FP16/AMP: độ chính xác ---
    fused = fuse_conv_bn(model).to(device)
    n_fused = getattr(fused, "n_fused_bn", 0)
    if n_fused:
        _, _, (zb,) = predict_views(fused, loader_for(val_df, args.images_dir, 224, True), device)
        add("I08a", f"gộp BN vào conv ({n_fused} cặp), FP32", aggregate_views([zb]), y, 1,
            max_abs_logit_diff=float(np.abs(zb - z0).max()))
    half = (fuse_conv_bn(model) if n_fused else copy.deepcopy(model)).to(device).half()
    if half is not None:
        import torch.nn as nn

        class HalfIn(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x.half())

        _, _, (zh,) = predict_views(HalfIn(half), loader_for(val_df, args.images_dir, 224, True), device)
        add("I08b", ("gộp BN + " if n_fused else "") + "FP16 (model.half())", aggregate_views([zh]), y, 1,
            max_abs_logit_diff=float(np.abs(zh - z0).max()))
    _, _, (za,) = predict_views(model, loader_for(val_df, args.images_dir, 224, True), device,
                                amp_dtype=torch.float16)
    add("I08c", "AMP (autocast fp16)", aggregate_views([za]), y, 1, max_abs_logit_diff=float(np.abs(za - z0).max()))

    # --- I09 precise-BN: tính lại thống kê BN bằng ảnh TRAIN qua transform val (không tốn thêm khi suy luận) ---
    from inference import recalibrate_bn
    train_df = D.load_split(args.labels_dir, 0)[0]
    if args.limit:
        train_df = train_df.head(args.limit)
    pbn = recalibrate_bn(model, loader_for(train_df, args.images_dir, 224, True), device)
    if getattr(pbn, "n_recalibrated_bn", 0):
        _, _, (zp, zpf) = predict_views(pbn, loader_for(val_df, args.images_dir, 224, True), device, views_hflip)
        add("I09", f"precise-BN ({pbn.n_recalibrated_bn} lớp BN, ảnh train), 1 view", aggregate_views([zp]), y)
        add("I09_I01", "precise-BN + TTA lật ngang", aggregate_views([zp, zpf], "prob"), y, 2)
        Tp = fit_temperature(np.mean([zp, zpf], 0), y)
        add("I09_I01_I07", f"precise-BN + TTA lật + temperature (T={Tp:.3f})",
            apply_temperature(np.mean([zp, zpf], 0), Tp), y, 2, T=Tp)
    del pbn

    # --- độ trễ (forward model, không tính tiền xử lý), batch 1 và batch 32 ---
    dev = device.type
    it = args.iters

    def L(key, m, size=224, dtype="fp32", forward=None, fused_bn=False):
        for b in (1, 32):
            r = latency_report(m, b, size, dtype, dev, warmup=10, iters=it if b == 1 else max(30, it // 3),
                               forward=forward, fused_bn=fused_bn)
            lat[f"{key}_b{b}"] = {"config": key, **r}
            print(f"  latency {key:22s} b{b:<3d} p50={r['p50']:.2f} p95={r['p95']:.2f} p99={r['p99']:.2f} ms "
                  f"{r['images_per_s']:.0f} img/s", flush=True)

    if args.skip_latency:
        L = lambda *a, **k: None  # noqa: E731
    L("I00_fp32", model)
    L("I01_hflip_K2", model, forward=lambda m, x: [m(v) for v in views_hflip(x)])
    L("I02b_10crop", model, size=256, forward=lambda m, x: [m(v) for v in views_multicrop(x, 224, flip=True)])
    L("I02c_3scale", model, size=256, forward=lambda m, x: [m(v) for v in views_multiscale(x, [224, 256, 288])])
    for s in (256, 288, 320):
        L(f"I04_res{s}", model, size=s)
    L("I08c_amp", model, dtype="amp")
    if n_fused:
        L("I08a_fusedBN_fp32", fused, fused_bn=True)
        L("I08b_fusedBN_fp16", fuse_conv_bn(model).to(device), dtype="fp16", fused_bn=True)
    else:
        L("I08b_fp16", model, dtype="fp16")
    if args.ensemble and not args.skip_latency:
        from model import build_model
        members = []
        for e in args.ensemble:
            cfg = json.loads((Path("runs") / e / "seed0" / "config.json").read_text())
            members.append(build_model(cfg["config"]["backbone"], pretrained=False).to(device).eval())
        ens_lat = {}
        for b in (1, 32):
            per = [latency_report(mm, b, 224, "fp32", dev, warmup=10, iters=50) for mm in members]
            ens_lat[b] = {"p50": sum(p["p50"] for p in per), "p95": sum(p["p95"] for p in per),
                          "p99": sum(p["p99"] for p in per)}
            lat[f"I05_ensemble_b{b}"] = {"config": "I05_ensemble", "gpu": per[0]["gpu"], "dtype": "fp32",
                                         "batch": b, "img_size": 224, "fused_bn": False, **ens_lat[b],
                                         "images_per_s": b / (ens_lat[b]["p50"] / 1000), "torch": per[0]["torch"],
                                         "note": "tổng độ trễ đo riêng từng thành viên (chạy tuần tự)"}
        print("  latency ensemble", ens_lat)

    out = {"exp": args.exp, "backbone": args.backbone, "seed": args.seed, "n_val": int(len(y)), "results": res}
    Path("logs").mkdir(exist_ok=True)
    Path(f"logs/step3_inference{args.out_suffix}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    Path(f"logs/step3_latency{args.out_suffix}.json").write_text(json.dumps(lat, indent=2, ensure_ascii=False))
    print("saved logs/step3_inference.json, logs/step3_latency.json")


if __name__ == "__main__":
    main()
