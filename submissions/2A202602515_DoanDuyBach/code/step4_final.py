"""Bước 4: chạy TEST đúng MỘT lần cho mỗi seed của cấu hình chung kết và của mốc (T00 + I00).

Chạy từ thư mục bài nộp (sau khi đã train F01 và T00 với các seed):
    python code/step4_final.py --exp F01 --backbone <tên> --seeds 0 1 2 --views none --eval-mode full --img-size 288 --temperature
    python code/step4_final.py --exp T00 --backbone <tên> --seeds 0 1 2 --views none --eval-mode crop --img-size 224
Mỗi (exp, seed) có file khoá runs/<exp>/seed<k>/test_done.json: script TỪ CHỐI chạy test lần hai.

Ghi ra predictions/:
  <exp>_seed<k>_test.csv   dự đoán test của cấu hình (đã temperature scaling nếu --temperature)
  <exp>_seed<k>_val.csv    dự đoán val với ĐÚNG cùng phương pháp suy luận (để chấm I4(b))
  <exp>uncal_seed<k>_test.csv  cùng cấu hình nhưng T = 1 (chỉ khi --temperature, để chấm I4(a))
Nhiệt độ T khớp trên VAL của chính seed đó (S2/S4), không nhìn test.
Gộp view: trung bình LOGIT rồi softmax (để một T duy nhất áp lên logit gộp).
Chung kết dùng: --eval-mode full --img-size 288 (ảnh đầy đủ resize 288, 1 view = I04_288) --temperature.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402
from eval import compute_metrics, save_predictions  # noqa: E402


def main():
    import dataset as D
    from inference import apply_temperature, fit_temperature, predict_views, recalibrate_bn, views_hflip
    from step3_inference import load_model, loader_for

    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--fold", type=int, default=0, help="fold của tác giả (0 cho bài chính; 1-4 chỉ cho điểm thưởng, S6)")
    ap.add_argument("--views", choices=["none", "hflip"], default="none")
    ap.add_argument("--temperature", action="store_true")
    ap.add_argument("--precise-bn", action="store_true", help="tính lại thống kê BN bằng ảnh train (I09)")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--eval-mode", choices=["crop", "full"], default="crop",
                    help="crop: Resize(size/0.875)+CenterCrop(size) (I00); full: ảnh đầy đủ resize về size (I04)")
    ap.add_argument("--dry-run", type=int, default=None,
                    help="CHẠY THỬ: dùng N ảnh VAL thay cho test, ghi vào --dry-out, không tạo file khoá, không đụng test")
    ap.add_argument("--dry-out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--images-dir", default="../../data/images")
    ap.add_argument("--labels-dir", default="../../data/labels")
    args = ap.parse_args()

    import torch
    import step3_inference

    device = torch.device(args.device) if args.device else TR.get_device()
    train_df, val_df, test_df = D.load_split(args.labels_dir, args.fold)
    pred_dir, log_dir = Path("predictions"), Path("logs")
    if args.dry_run:
        val_df = val_df.head(args.dry_run)
        test_df = val_df.copy()  # KHÔNG dùng test khi chạy thử
        pred_dir = log_dir = Path(args.dry_out)
        step3_inference.NUM_WORKERS = 0
    views_fn = views_hflip if args.views == "hflip" else None
    crop = args.eval_mode == "crop"
    log = {}
    for seed in args.seeds:
        rd = Path("runs") / args.exp / f"seed{seed}"
        lock = rd / "test_done.json" if not args.dry_run else Path(args.dry_out) / f"{args.exp}_{seed}_lock.json"
        if lock.exists():
            print(f"{args.exp} seed{seed}: test ĐÃ chạy ({lock}); không chạy lại.")
            log[seed] = json.loads(lock.read_text())
            continue
        model = load_model(args.exp, args.backbone, seed, device)
        if args.precise_bn:
            model = recalibrate_bn(model, loader_for(train_df, args.images_dir, args.img_size, True), device)

        # VAL: cùng phương pháp suy luận, khớp T
        vn, yv, zv = predict_views(model, loader_for(val_df, args.images_dir, args.img_size, crop), device, views_fn)
        zv = np.mean(zv, 0)
        T = fit_temperature(zv, yv) if args.temperature else 1.0
        pv = apply_temperature(zv, T)
        save_predictions(pred_dir / f"{args.exp}_seed{seed}_val.csv", vn, yv, pv)

        # TEST: đúng một lần
        tn, yt, zt = predict_views(model, loader_for(test_df, args.images_dir, args.img_size, crop), device, views_fn)
        zt = np.mean(zt, 0)
        if not args.dry_run:
            np.save(rd / "test_logits_final.npy", zt)
            (rd / "test_names.json").write_text(json.dumps(tn))
        pt = apply_temperature(zt, T)
        save_predictions(pred_dir / f"{args.exp}_seed{seed}_test.csv", tn, yt, pt)
        if args.temperature:
            save_predictions(pred_dir / f"{args.exp}uncal_seed{seed}_test.csv", tn, yt,
                             apply_temperature(zt, 1.0))
        mv = compute_metrics(yv, pv.argmax(1), pv)
        mt = compute_metrics(yt, pt.argmax(1), pt)
        rec = {"exp": args.exp, "seed": seed, "fold": args.fold, "views": args.views, "eval_mode": args.eval_mode, "img_size": args.img_size, "precise_bn": args.precise_bn, "temperature": T,
               "val_macro_f1": mv["macro_f1"], "test_macro_f1": mt["macro_f1"], "test_top1": mt["top1"],
               "test_ece": mt["ece"], "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        lock.write_text(json.dumps(rec, indent=2))
        log[seed] = rec
        print(json.dumps(rec), flush=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / f"step4_{args.exp}.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
