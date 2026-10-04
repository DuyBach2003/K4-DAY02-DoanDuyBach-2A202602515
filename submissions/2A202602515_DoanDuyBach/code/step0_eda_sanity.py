"""Bước 0: EDA, kiểm tra chia dữ liệu (README mục 2.1) và kiểm tra pipeline (GUIDE mục 1.3).

Chạy từ thư mục bài nộp:
    python code/step0_eda_sanity.py --images-dir ../../data/images --labels-dir ../../data/labels
Kết quả: figures/eda_*.png, figures/sanity_*.png, logs/step0.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dataset as D  # noqa: E402

PAPER_TABLE1 = {"Chinee Apple": 1125, "Lantana": 1064, "Parkinsonia": 1031, "Parthenium": 1022,
                "Prickly Acacia": 1062, "Rubber Vine": 1009, "Siam Weed": 1074, "Snake Weed": 1016,
                "Negatives": 9106}


def eda(args, out: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    tr, va, te = D.load_split(args.labels_dir, 0)
    info = D.check_split(tr, va, te, args.images_dir)
    out["split_check"] = info

    # bảng đếm theo lớp + đối chiếu Table 1
    rows = []
    for c, name in enumerate(D.CLASS_NAMES):
        n_tr, n_va, n_te = (info["per_class"][k][c] for k in ("train", "val", "test"))
        rows.append({"class": name, "train": n_tr, "val": n_va, "test": n_te, "total": n_tr + n_va + n_te,
                     "paper_table1": PAPER_TABLE1[name], "diff_vs_paper": n_tr + n_va + n_te - PAPER_TABLE1[name]})
    out["per_class_table"] = rows
    totals = [r["total"] for r in rows]
    out["imbalance_ratio_max_min"] = max(totals) / min(totals)
    out["negatives_share"] = totals[-1] / sum(totals)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(9)
    w = 0.27
    for i, k in enumerate(("train", "val", "test")):
        axes[0].bar(x + (i - 1) * w, [r[k] for r in rows], w, label=k)
    axes[0].set_xticks(x, D.CLASS_NAMES, rotation=35, ha="right")
    axes[0].set_ylabel("số ảnh"); axes[0].set_title("Phân bố lớp theo tập (fold 0)"); axes[0].legend()
    axes[0].set_yscale("log")
    axes[1].bar(x - 0.2, totals, 0.4, label="đếm thật (train+val+test)")
    axes[1].bar(x + 0.2, [r["paper_table1"] for r in rows], 0.4, label="Table 1 bài báo")
    axes[1].set_xticks(x, D.CLASS_NAMES, rotation=35, ha="right")
    axes[1].set_ylabel("số ảnh"); axes[1].set_title(f"Đối chiếu Table 1 (tỉ lệ max/min = "
                                                    f"{out['imbalance_ratio_max_min']:.2f})")
    axes[1].legend()
    fig.tight_layout(); fig.savefig(args.fig_dir / "eda_class_distribution.png", dpi=130); plt.close(fig)

    # ảnh mẫu: 4 ảnh mỗi lớp từ train
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(9, 4, figsize=(8, 18))
    for c in range(9):
        names = tr[tr["Label"] == c]["Filename"].to_numpy()
        for j, f in enumerate(rng.choice(names, 4, replace=False)):
            ax = axes[c, j]
            ax.imshow(Image.open(Path(args.images_dir) / f).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(D.CLASS_NAMES[c], fontsize=9)
    fig.suptitle("Ảnh mẫu (train, 4 ảnh/lớp)")
    fig.tight_layout(); fig.savefig(args.fig_dir / "eda_samples.png", dpi=90); plt.close(fig)

    # thống kê kích thước/kênh + mean/std pixel trên mẫu 500 ảnh train
    sizes, modes, px = set(), set(), []
    for f in rng.choice(tr["Filename"].to_numpy(), 500, replace=False):
        im = Image.open(Path(args.images_dir) / f)
        sizes.add(im.size); modes.add(im.mode)
        px.append(np.asarray(im.convert("RGB"), dtype=np.float32).reshape(-1, 3) / 255.0)
    px = np.concatenate(px)
    out["image_stats"] = {"sizes": sorted(map(list, sizes)), "modes": sorted(modes),
                          "train_sample_mean": px.mean(0).round(4).tolist(),
                          "train_sample_std": px.std(0).round(4).tolist(),
                          "imagenet_mean": list(D.IMAGENET_MEAN), "imagenet_std": list(D.IMAGENET_STD)}
    print(json.dumps(out["image_stats"]))
    return tr, va, te


def sanity(args, out: dict, tr):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn

    import train as TR
    from losses import mix_batch
    from model import build_model

    TR.set_seed(0)
    device = TR.get_device()

    # (4) ảnh sau augmentation, đã giải chuẩn hoá, kèm nhãn
    mean = torch.tensor(D.IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(D.IMAGENET_STD).view(3, 1, 1)
    sub = tr.groupby("Label").head(2).reset_index(drop=True)
    fig, axes = plt.subplots(3, 6, figsize=(14, 7.5))
    for row, aug in enumerate(("basic", "color", "trivial")):
        ds = D.DeepWeedsDataset(sub, args.images_dir, D.build_transforms(True, 224, aug))
        for j in range(6):
            x, y, f = ds[j * 3 % len(ds)]
            axes[row, j].imshow((x * std + mean).clamp(0, 1).permute(1, 2, 0))
            axes[row, j].set_title(f"{aug}: {D.CLASS_NAMES[y]}", fontsize=8)
            axes[row, j].axis("off")
    fig.suptitle("Ảnh sau augmentation (đã giải chuẩn hoá) và nhãn")
    fig.tight_layout(); fig.savefig(args.fig_dir / "sanity_augmentations.png", dpi=100); plt.close(fig)

    # CutMix / Mixup trực quan + lam
    loader = D.make_loader(sub, args.images_dir, D.build_transforms(True, 224, "basic"), 8, True, None, 0)
    x, y, _ = next(iter(loader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5))
    for r, mode in enumerate(("cutmix", "mixup")):
        xm, (ya, yb, lam) = mix_batch(x, y, 1.0, mode, np.random.default_rng(r))
        for j in range(4):
            axes[r, j].imshow((xm[j] * std + mean).clamp(0, 1).permute(1, 2, 0))
            axes[r, j].set_title(f"{mode} λ={lam:.2f}\n{D.CLASS_NAMES[ya[j]]} / {D.CLASS_NAMES[yb[j]]}", fontsize=8)
            axes[r, j].axis("off")
    fig.tight_layout(); fig.savefig(args.fig_dir / "sanity_mix.png", dpi=100); plt.close(fig)

    # (2) loss ban đầu ≈ ln 9 trên 1 batch val-transform, head mới, eval mode
    model = build_model("resnet50", True, 9).to(device)
    eval_loader = D.make_loader(tr.sample(256, random_state=0), args.images_dir,
                                D.build_transforms(False, 224), 64, False, None, 0)
    model.eval()
    losses = []
    with torch.inference_mode():
        for xb, yb, _ in eval_loader:
            losses.append(nn.functional.cross_entropy(model(xb.to(device)).float(), yb.to(device)).item())
    out["initial_loss"] = {"resnet50_pretrained_new_head": float(np.mean(losses)), "ln9": math.log(9)}
    print("initial loss", out["initial_loss"])

    # (3) overfit một batch nhỏ (16 ảnh, không augmentation) tới loss gần 0
    TR.set_seed(0)
    small = tr.groupby("Label").head(2).sample(16, random_state=0)
    lb = D.make_loader(small, args.images_dir, D.build_transforms(False, 224), 16, False, None, 0)
    xb, yb, _ = next(iter(lb))
    xb, yb = xb.to(device), yb.to(device)
    model = build_model("resnet50", True, 9).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3 * 0.3)
    curve = []
    model.train()
    for step in range(60):
        loss = nn.functional.cross_entropy(model(xb), yb)
        opt.zero_grad(); loss.backward(); opt.step()
        curve.append(loss.item())
    model.eval()
    with torch.inference_mode():
        acc = (model(xb).argmax(1) == yb).float().mean().item()
    out["overfit_one_batch"] = {"n_images": 16, "steps": 60, "loss_first": curve[0], "loss_last": curve[-1],
                                "train_acc_eval_mode": acc}
    print("overfit", out["overfit_one_batch"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(curve); ax.axhline(math.log(9), ls=":", color="gray", label="ln 9")
    ax.set_xlabel("bước"); ax.set_ylabel("loss (log)"); ax.set_title("Overfit 1 batch 16 ảnh (ResNet-50)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(args.fig_dir / "sanity_overfit_one_batch.png", dpi=120); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="../../data/images")
    ap.add_argument("--labels-dir", default="../../data/labels")
    ap.add_argument("--fig-dir", default="figures", type=Path)
    ap.add_argument("--out", default="logs/step0.json", type=Path)
    args = ap.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out: dict = {}
    tr, _, _ = eda(args, out)
    sanity(args, out, tr)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print("saved", args.out)


if __name__ == "__main__":
    main()
