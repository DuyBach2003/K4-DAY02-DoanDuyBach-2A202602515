"""Biểu đồ phân tích cho báo cáo (từ log/dự đoán thật, không nhập số tay).

Chạy từ thư mục bài nộp:
    python code/make_figures.py --images-dir ../../data/images --labels-dir ../../data/labels
Kết quả trong figures/:
  backbones_f1_vs_latency.png   macro-F1 val theo độ trễ batch-1 (kích thước điểm ~ GMAC)
  training_ablation.png         Δ macro-F1 val của từng ablation so với T00
  inference_tradeoff.png        macro-F1 val theo độ trễ p50 batch-1 của các phương pháp suy luận
  reliability_I00_vs_I07.png    biểu đồ độ tin cậy trước/sau temperature scaling (val)
  confusion_test_<exp>.png      ma trận nhầm lẫn test (tổng các seed) của chung kết và mốc
Ngoài ra vẽ lại curves/T00_*.png và curves/F01_*.png với đủ 3 seed trên một ảnh.
  errors_chinee_snake.png       ảnh test bị nhầm giữa Chinee Apple và Snake Weed (chung kết, seed 0)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import train  # noqa: E402,F401  (đặt sys.path tới eval.py gốc)
from experiments import REGISTRY, TRAINING_META  # noqa: E402

CLASS = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed",
         "Snake Weed", "Negatives"]
FIG = Path("figures")


def run_summary(exp, seed=0):
    p = Path("logs") / exp / f"seed{seed}" / "summary.json"
    return json.loads(p.read_text()) if p.exists() else None


def fig_backbones():
    rows = []
    for exp in sorted(e for e in REGISTRY if e.startswith("B")):
        s = run_summary(exp)
        if s:
            rows.append((exp, REGISTRY[exp]["desc"], s["val_macro_f1"], s["latency_b1_fp32"]["p50"], s["gmacs"],
                         s["params_m"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for exp, name, f1, lat, g, p in rows:
        ax.scatter(lat, f1, s=60 + 60 * g, alpha=.7)
        ax.annotate(f"{exp} {name}\n{p:.1f}M, {g:.2f} GMAC", (lat, f1), textcoords="offset points", xytext=(6, -4),
                    fontsize=8)
    ax.set_xlabel("độ trễ batch 1, FP32, p50 (ms) - Apple M5 MPS")
    ax.set_ylabel("macro-F1 val")
    ax.set_title("Bước 1: backbone (cùng công thức nền, 10 epoch, seed 0)\nkích thước điểm ~ GMAC")
    ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIG / "backbones_f1_vs_latency.png", dpi=130); plt.close(fig)


def fig_training():
    base = run_summary("T00")
    if not base:
        return
    rows = [(e, TRAINING_META[e]["diff"], run_summary(e)) for e in TRAINING_META if e != "T00"]
    rows = [(e, d, s) for e, d, s in rows if s]
    if not rows:
        return
    rows.sort(key=lambda r: r[2]["val_macro_f1"])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.5))
    deltas = [s["val_macro_f1"] - base["val_macro_f1"] for _, _, s in rows]
    ax.barh([f"{e}: {d[:55]}" for e, d, _ in rows], deltas, color=["tab:green" if v > 0 else "tab:red" for v in deltas])
    for i, v in enumerate(deltas):
        ax.text(v, i, f" {v:+.4f}", va="center", fontsize=8)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel(f"Δ macro-F1 val so với T00 ({base['val_macro_f1']:.4f}); 1 seed")
    ax.set_title("Bước 2: ablation công thức huấn luyện (DeiT-S)")
    ax.grid(alpha=.3, axis="x")
    fig.tight_layout(); fig.savefig(FIG / "training_ablation.png", dpi=130); plt.close(fig)


def fig_inference():
    p = Path("logs/step3_inference.json")
    if not p.exists():
        return
    import make_results
    inf, _ = make_results.sheet_inference()
    d = inf.dropna(subset=["độ trễ p50 b1 (ms)"])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, r in d.iterrows():
        ax.scatter(r["độ trễ p50 b1 (ms)"], r["macro-F1 val"], s=40)
        ax.annotate(r["exp_id"], (r["độ trễ p50 b1 (ms)"], r["macro-F1 val"]), textcoords="offset points",
                    xytext=(4, 3), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("độ trễ batch 1, p50 (ms, thang log) - Apple M5 MPS")
    ax.set_ylabel("macro-F1 val")
    ax.set_title("Bước 3: đánh đổi độ chính xác - độ trễ của các phương pháp suy luận")
    ax.axvline(100, color="red", ls=":", label="ngân sách thời gian thực 100 ms")
    ax.legend(); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(FIG / "inference_tradeoff.png", dpi=130); plt.close(fig)


def reliability(ax, probs, y, title, bins=15):
    conf = probs.max(1)
    corr = probs.argmax(1) == y
    idx = np.clip(np.ceil(conf * bins).astype(int) - 1, 0, bins - 1)
    accs, confs, ns = [], [], []
    for m in range(bins):
        k = idx == m
        if k.any():
            accs.append(corr[k].mean()); confs.append(conf[k].mean()); ns.append(k.sum())
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    ax.scatter(confs, accs, s=np.array(ns) / max(ns) * 200 + 10)
    ax.plot(confs, accs, alpha=.6)
    from eval import ece_score
    ax.set_title(f"{title}\nECE = {ece_score(probs, y):.4f}")
    ax.set_xlabel("độ tin cậy (max softmax)"); ax.set_ylabel("accuracy"); ax.grid(alpha=.3)


def fig_reliability(exp="T00"):
    rd = Path("runs") / exp / "seed0"
    if not (rd / "val_logits.npy").exists() or not Path("logs/step3_inference.json").exists():
        return
    from inference import apply_temperature
    z, y = np.load(rd / "val_logits.npy"), np.load(rd / "val_labels.npy")
    T = json.loads(Path("logs/step3_inference.json").read_text())["results"]["I07_I00"]["T"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    reliability(axes[0], apply_temperature(z, 1.0), y, f"{exp} I00 (T = 1), val")
    reliability(axes[1], apply_temperature(z, T), y, f"{exp} I07 (T = {T:.3f}, khớp trên val), val")
    fig.tight_layout(); fig.savefig(FIG / "reliability_I00_vs_I07.png", dpi=130); plt.close(fig)


def fig_seed_curves(exp, seeds=(0, 1, 2)):
    """Ảnh biểu đồ training của thí nghiệm nhiều seed (T00, F01): mọi seed trên một ảnh, ghi đè ảnh của train.py
    (train.py vẽ theo từng lần chạy nên ảnh chỉ còn seed chạy sau cùng)."""
    hs = {s: pd.read_csv(p) for s in seeds if (p := Path("logs") / exp / f"seed{s}" / "history.csv").exists()}
    if len(hs) < 2:
        return
    cfg = REGISTRY[exp]  # không đọc config.json của seed 0: đó là bản sao của B03/T09 (exp_id, desc khác)
    lr = pd.read_csv(Path("logs") / exp / f"seed{min(hs)}" / "lr.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for i, (s, h) in enumerate(hs.items()):
        c = f"C{i}"
        best = h.sort_values(["val_macro_f1", "epoch"], ascending=[False, True]).iloc[0]
        axes[0].plot(h.epoch, h.train_loss, "o-", color=c, label=f"seed {s}: train loss")
        axes[0].plot(h.epoch, h.val_loss, "s--", color=c, label=f"seed {s}: val loss (CE)")
        axes[1].plot(h.epoch, h.val_macro_f1, "o-", color=c,
                     label=f"seed {s}: val macro-F1 (best ep {int(best.epoch)}: {best.val_macro_f1:.4f})")
        axes[1].plot(h.epoch, h.val_top1, "s:", color=c, alpha=.6, label=f"seed {s}: val top-1")
    axes[0].set_title("Loss"); axes[0].set_ylabel("loss")
    axes[1].set_title("Metric val"); axes[1].set_ylabel("metric"); axes[1].legend(fontsize=7, loc="lower right")
    axes[0].legend(fontsize=7)
    for a in axes[:2]:
        a.set_xlabel("epoch"); a.grid(alpha=.3); a.set_xticks(range(1, int(max(len(h) for h in hs.values())) + 1))
    axes[2].plot(lr.step, lr.lr)
    axes[2].set_xlabel("bước (iteration)"); axes[2].set_ylabel("LR (nhóm lớn nhất)")
    axes[2].set_title(f"Lịch LR (seed {min(hs)}; các seed giống nhau)"); axes[2].grid(alpha=.3)
    fig.suptitle(f"{exp} · {cfg['backbone']} · {len(hs)} seed ({', '.join(map(str, hs))}) · {cfg['desc']}")
    fig.tight_layout()
    fig.savefig(Path("curves") / f"{exp}_{cfg['desc']}.png", dpi=130); plt.close(fig)


def fig_bonus_shift():
    """Điểm thưởng: macro-F1 và ECE dưới lệch phân phối (val), và test-time adaptation (logs/bonus_*.json)."""
    p, q = Path("logs/bonus_shift.json"), Path("logs/bonus_tta.json")
    if not p.exists():
        return
    from make_results import SHIFT_NAMES

    d = json.loads(p.read_text())
    conds = list(d["conditions"])
    x = np.arange(len(conds))
    tta = json.loads(q.read_text()) if q.exists() else {}
    fig, axes = plt.subplots(1, 3 if tta else 2, figsize=(19 if tta else 13, 4.8))
    ax = axes[0]
    for k, (exp, lab) in enumerate((("F01", "F01 chung kết (ảnh đầy đủ 288)"), ("T00", "T00 mốc (CenterCrop 224)"))):
        s = [d["results"][exp][c]["summary"]["macro_f1"] for c in conds]
        ax.bar(x + (k - 0.5) * 0.38, [v["mean"] for v in s], 0.38, yerr=[v["std"] for v in s], capsize=3, label=lab)
    ax.set_ylim(0, 1); ax.set_ylabel("macro-F1 val (mean ± std, 3 seed)"); ax.set_title("Độ chính xác dưới lệch phân phối")
    ax = axes[1]
    for k, (key, lab) in enumerate((("ece_T1", "T = 1"), ("ece_Tclean", "T khớp trên val sạch"),
                                    ("ece_Toracle", "T oracle (khớp lại trên ảnh lệch)"))):
        ax.bar(x + (k - 1) * 0.27, [d["results"]["F01"][c]["summary"][key]["mean"] for c in conds], 0.27, label=lab)
    ax.set_ylabel("ECE val (15 bin), F01, TB 3 seed"); ax.set_title("Hiệu chuẩn của F01 dưới lệch phân phối")
    if tta:
        ax = axes[2]
        shifted = [c for c in conds if c != "clean"]
        xs = np.arange(len(shifted))
        series = []
        if "B06" in tta.get("bn", {}).get("results", {}):
            r = tta["bn"]["results"]["B06"]
            series += [("B06 trước", [r[c]["goc"]["macro_f1"] for c in shifted]),
                       ("B06 + BN-adapt", [r[c]["bn_adapt"]["macro_f1"] for c in shifted])]
        if tta.get("tent", {}).get("results"):
            r = tta["tent"]["results"]
            series += [("F01 trước", [r[c]["truoc"]["macro_f1"] for c in shifted]),
                       ("F01 + Tent", [r[c]["tent"]["macro_f1"] for c in shifted])]
        w = 0.8 / max(len(series), 1)
        for k, (lab, v) in enumerate(series):
            ax.bar(xs + (k - (len(series) - 1) / 2) * w, v, w, label=lab)
        ax.set_xticks(xs, [SHIFT_NAMES[c] for c in shifted], rotation=30, ha="right")
        ax.set_ylim(0, 1.2); ax.set_yticks(np.arange(0, 1.01, 0.2)); ax.set_ylabel("macro-F1 val")
        ax.set_title("Test-time adaptation (không dùng nhãn)")
        ax.legend(fontsize=8, ncol=2, loc="upper center"); ax.grid(alpha=.3, axis="y")
    for ax in axes[:2]:
        ax.set_xticks(x, [SHIFT_NAMES[c] for c in conds], rotation=30, ha="right")
        ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.suptitle("Điểm thưởng: lệch phân phối tự tạo trên VAL fold 0 (không dùng test)")
    fig.tight_layout(); fig.savefig(FIG / "bonus_shift_tta.png", dpi=130); plt.close(fig)


def fig_confusion(tag):
    p = Path(f"eval_out/{tag}_confusion_sum.csv")
    if not p.exists():
        return
    cm = pd.read_csv(p, index_col=0).to_numpy()
    rown = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.imshow(rown, cmap="Blues", vmin=0, vmax=1)
    for i in range(9):
        for j in range(9):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}\n{rown[i, j]:.1%}", ha="center", va="center", fontsize=7,
                        color="white" if rown[i, j] > .5 else "black")
    ax.set_xticks(range(9), CLASS, rotation=40, ha="right"); ax.set_yticks(range(9), CLASS)
    ax.set_xlabel("dự đoán"); ax.set_ylabel("nhãn thật")
    ax.set_title(f"Ma trận nhầm lẫn TEST - {tag} (tổng các seed; số ảnh và % theo hàng)")
    fig.tight_layout(); fig.savefig(FIG / f"confusion_test_{tag}.png", dpi=130); plt.close(fig)


def fig_errors(images_dir, exp="F01", seed=0, n=8):
    p = Path(f"predictions/{exp}_seed{seed}_test.csv")
    if not p.exists():
        return
    from PIL import Image
    d = pd.read_csv(p)
    pairs = [(0, 7), (7, 0)]
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 5))
    out = {}
    for r, (t, pr) in enumerate(pairs):
        e = d[(d.y_true == t) & (d.y_pred == pr)].copy()
        e["conf"] = e[[f"p{pr}"]].max(1)
        e = e.sort_values("conf", ascending=False)
        out[f"{CLASS[t]} -> {CLASS[pr]}"] = int(len(e))
        for j in range(n):
            ax = axes[r, j]; ax.axis("off")
            if j < len(e):
                row = e.iloc[j]
                ax.imshow(Image.open(Path(images_dir) / row.Filename).convert("RGB"))
                ax.set_title(f"thật {CLASS[t][:7]}\nđoán {CLASS[pr][:7]} {row.conf:.2f}", fontsize=7)
    fig.suptitle(f"Ảnh TEST bị nhầm Chinee Apple ↔ Snake Weed ({exp} seed {seed}); số ca: {out}", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "errors_chinee_snake.png", dpi=110); plt.close(fig)
    # các cặp nhầm nhiều nhất
    cm = pd.crosstab(d.y_true, d.y_pred)
    top = [(CLASS[i], CLASS[j], int(cm.loc[i, j])) for i in cm.index for j in cm.columns if i != j and cm.loc[i, j]]
    top.sort(key=lambda x: -x[2])
    Path("logs/error_pairs.json").write_text(json.dumps({"exp": exp, "seed": seed, "top_pairs": top[:10]},
                                                        ensure_ascii=False, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", default="../../data/images")
    ap.add_argument("--labels-dir", default="../../data/labels")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    fig_backbones()
    fig_training()
    fig_inference()
    fig_reliability()
    for exp in ("T00", "F01"):
        fig_seed_curves(exp)
    fig_bonus_shift()
    for tag in ("F01", "T00"):
        fig_confusion(tag)
    fig_errors(args.images_dir)
    print("saved figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
