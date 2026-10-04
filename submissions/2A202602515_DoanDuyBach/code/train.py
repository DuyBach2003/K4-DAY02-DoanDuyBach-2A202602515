"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Dùng MỘT hàm `run(cfg)` cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi `Config`.

Chạy một thí nghiệm từ dòng lệnh:
    python train.py --set exp_id=B01 backbone=resnet50 seed=0
Chỉ số dùng để chọn checkpoint (macro-F1 val) tính bằng eval.compute_metrics của repo gốc.

Quy ước:
  - LR cập nhật theo BƯỚC (iteration): warmup tuyến tính `warmup_epochs` rồi cosine về 0.
  - Loss val luôn là cross-entropy thường (so sánh được giữa các thí nghiệm dùng loss khác nhau).
  - Checkpoint tốt nhất = epoch có macro-F1 val cao nhất (hòa: epoch sớm hơn). Có EMA thì đánh giá và lưu
    trọng số EMA.
  - Test KHÔNG được đụng tới trong run() trừ khi cfg.save_test_predictions=True (chỉ ở Bước 4).
  - Mỗi epoch lưu runs/<exp>/seed<k>/last.pt; chạy lại cùng cấu hình sẽ TIẾP TỤC từ epoch kế tiếp (thứ tự batch
    sau khi resume không trùng hoàn toàn với lần chạy liền mạch). last.pt bị xoá khi run xong.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    """Thư mục chứa eval.py gốc (đi ngược lên từ code/)."""
    for p in [CODE_DIR, *CODE_DIR.parents]:
        if (p / "eval.py").exists() and p != CODE_DIR:
            return p
    raise FileNotFoundError("không tìm thấy eval.py của repo gốc")


REPO_ROOT = _find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CODE_DIR))

from eval import compute_metrics, save_predictions  # noqa: E402


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug | flipv
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    grad_clip: float | None = None
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"             # config.json, history.csv, checkpoint, logit của từng lần chạy
    pred_dir: str = "predictions"     # file dự đoán đúng định dạng eval.py (nộp cùng bài)
    curves_dir: str = "curves"
    desc: str = ""                    # mô tả ngắn, dùng cho tên ảnh curves/<exp_id>_<desc>.png
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False
    # --- tiện ích ---
    measure_latency: bool = True      # đo độ trễ sơ bộ batch 1 sau khi train (Bước 1)
    max_train_batches: int | None = None  # chỉ để chạy thử nhanh / kiểm tra pipeline


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn của file dự đoán: <pred_dir>/<exp_id>_seed<k>_<split>.csv (split = val | test)."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    suffix = f"_{cfg.desc}" if cfg.desc else ""
    return Path(cfg.curves_dir) / f"{cfg.exp_id}{suffix}.png"


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_dtype(device):
    """dtype cho autocast: fp16 + GradScaler trên CUDA và MPS, None (fp32) trên CPU.

    Đã đo trên Apple M5 (MPS, ResNet-50, batch 64): fp32 43 ảnh/s, bf16 35 ảnh/s, fp16 53 ảnh/s -> chọn fp16.
    """
    import torch

    if device.type in ("cuda", "mps"):
        return torch.float16
    return None


def set_seed(seed: int) -> None:
    """Cố định random, numpy, torch (CPU/CUDA/MPS). DataLoader worker được seed qua dataset.seed_worker.

    Mức tái lập: trên CUDA bật cudnn.deterministic; trên MPS một số kernel (scatter/atomic) không tất định,
    nên hai lần chạy cùng seed có thể lệch nhỏ ở chữ số thập phân thứ 3-4 (ghi trong báo cáo).
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def build_optimizer(model, cfg: Config):
    """AdamW với các nhóm tham số của model.param_groups (norm/bias không weight decay)."""
    import torch

    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999))


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Hệ số nhân LR tại bước `step`: warmup tuyến tính (1/w ... 1) rồi cosine từ 1 về 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine về 0, cập nhật theo BƯỚC (scheduler.step() sau mỗi optimizer.step())."""
    import torch

    total = cfg.epochs * steps_per_epoch
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warm))


class EMA:
    """Trung bình động trọng số: W_ema <- d * W_ema + (1 - d) * W  (slide trang 56).

    - Giữ một bản sao riêng `self.module` để đánh giá (không đụng tới model đang train).
    - Decay khởi động: d_t = min(decay, (1 + t) / (10 + t)) để các bước đầu không bị trọng số khởi tạo kéo lại.
    - Buffer (BatchNorm running_mean/var, num_batches_tracked) được SAO CHÉP từ model đang train thay vì lấy
      trung bình: thống kê BN của model hiện tại là ước lượng hợp lý cho trọng số EMA khi EMA đã bám sát.
    """

    def __init__(self, model, decay: float):
        self.decay = decay
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    def update(self, model) -> None:
        import torch

        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        with torch.no_grad():
            for e, m in zip(self.module.parameters(), model.parameters()):
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            for e, m in zip(self.module.buffers(), model.buffers()):
                e.copy_(m)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, mix_rng=None) -> dict:
    """Một epoch huấn luyện. Trả về {"train_loss", "train_acc" (NaN nếu dùng mix), "lrs" (LR head theo bước)}."""
    import torch

    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)  # backbone đóng băng -> phần backbone ở eval (BN không cập nhật)
    dtype = amp_dtype(device) if cfg.amp else None
    tot_loss, tot_correct, tot_n, lrs = 0.0, 0, 0, []
    for i, (x, y, _) in enumerate(loader):
        if cfg.max_train_batches is not None and i >= cfg.max_train_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        targets = None
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix, generator=mix_rng)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
            logits = model(x)
        logits = logits.float()
        loss = mixed_loss(criterion, logits, targets) if targets is not None else criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss không hữu hạn ở bước {i}: {loss.item()}")

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        scheduler.step()
        if ema is not None:
            ema.update(model)

        bs = y.shape[0]
        tot_loss += loss.item() * bs
        tot_n += bs
        if targets is None:
            tot_correct += (logits.argmax(1) == y).sum().item()
        lrs.append(max(g["lr"] for g in optimizer.param_groups))
    return {"train_loss": tot_loss / max(tot_n, 1),
            "train_acc": float("nan") if cfg.mix else tot_correct / max(tot_n, 1),
            "lrs": lrs}


def evaluate(model, loader, criterion, device, amp: bool = False):
    """Chạy model ở chế độ eval, không gradient. Trả về (filenames, y_true[N], logits[N, 9], loss)."""
    import torch

    model.eval()
    dtype = amp_dtype(device) if amp else None
    names, ys, outs = [], [], []
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=dtype is not None):
                logits = model(x)
            outs.append(logits.float().cpu())
            ys.append(y)
            names.extend(f)
    logits = torch.cat(outs)
    y_true = torch.cat(ys)
    loss = float(criterion(logits, y_true).item()) if criterion is not None else float("nan")
    return names, y_true.numpy(), logits.numpy(), loss


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lrs: list[float] | None = None) -> None:
    """Ảnh 3 panel: loss train/val theo epoch; macro-F1/top-1 val (+ acc train) theo epoch; LR theo bước."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3 if lrs else 2, figsize=(16 if lrs else 11, 4.2))
    ax = axes[0]
    ax.plot(ep, [h["train_loss"] for h in history], "o-", label="train loss (loss huấn luyện)")
    ax.plot(ep, [h["val_loss"] for h in history], "s-", label="val loss (CE)")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Loss"); ax.grid(alpha=.3); ax.legend()

    ax = axes[1]
    ax.plot(ep, [h["val_macro_f1"] for h in history], "o-", label="val macro-F1")
    ax.plot(ep, [h["val_top1"] for h in history], "s-", label="val top-1")
    tr = [h["train_acc"] for h in history]
    if not all(np.isnan(tr)):
        ax.plot(ep, tr, "^--", label="train top-1 (đang augment)")
    best = max(history, key=lambda h: (h["val_macro_f1"], -h["epoch"]))
    ax.axvline(best["epoch"], color="gray", ls=":", label=f"best ep {best['epoch']}: F1={best['val_macro_f1']:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("metric"); ax.set_title("Metric"); ax.grid(alpha=.3)
    ax.legend(loc="lower right")

    for a in axes[:2]:
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
    if lrs:
        ax = axes[2]
        ax.plot(np.arange(1, len(lrs) + 1), lrs)
        ax.set_xlabel("bước (iteration)"); ax.set_ylabel("LR (nhóm lớn nhất)"); ax.set_title("Lịch LR")
        ax.grid(alpha=.3)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def lib_versions() -> dict:
    import timm
    import torch
    import torchvision

    return {"python": platform.python_version(), "torch": torch.__version__,
            "torchvision": torchvision.__version__, "timm": timm.__version__, "numpy": np.__version__,
            "platform": platform.platform(), "machine": platform.machine()}


def build_loss_for(cfg: Config, train_df):
    from dataset import NUM_CLASSES
    from losses import build_criterion, class_weights

    if cfg.loss == "ce_weighted":
        counts = train_df["Label"].value_counts().reindex(range(NUM_CLASSES)).to_numpy()
        w = class_weights(counts, beta=cfg.class_weight_beta or 0.0)
        return build_criterion("ce_weighted", weight=w), w.tolist()
    if cfg.loss == "ls":
        return build_criterion("ls", smoothing=cfg.label_smoothing or 0.1), None
    if cfg.loss == "focal":
        return build_criterion("focal", gamma=cfg.focal_gamma), None
    return build_criterion(cfg.loss), None


def run(cfg: Config) -> dict:
    """Huấn luyện một cấu hình và lưu mọi thứ cần thiết. Trả về dict kết quả tóm tắt.

    Lưu ở run_dir(cfg): config.json, history.csv, best.pt, val_logits.npy, val_names.json, summary.json;
    predictions/<exp_id>_seed<k>_val.csv; curves/<exp_id>_<desc>.png.
    """
    import pandas as pd
    import torch
    import torch.nn as nn

    import dataset as D
    from model import build_model, count_gmacs, count_params, weight_tag

    # 1. seed, thư mục, cấu hình
    set_seed(cfg.seed)
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    device = get_device()

    # 2. chia dữ liệu + kiểm tra S1-S4
    train_df, val_df, test_df = D.load_split(cfg.labels_dir, cfg.fold)
    split_info = D.check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False)

    # 3. loader (test loader chỉ tạo khi save_test_predictions)
    tf_train = D.build_transforms(True, cfg.img_size, cfg.aug)
    tf_eval = D.build_transforms(False, cfg.img_size)
    train_loader = D.make_loader(train_df, cfg.images_dir, tf_train, cfg.batch_size, True,
                                 cfg.sampler, cfg.num_workers, seed=cfg.seed)
    val_loader = D.make_loader(val_df, cfg.images_dir, tf_eval, 128, False, None, cfg.num_workers)

    # 4. model, loss, optimizer, scheduler, scaler, EMA
    model = build_model(cfg.backbone, True, D.NUM_CLASSES, cfg.drop_rate, cfg.init)
    params_m = count_params(model)
    trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    try:
        gmacs = count_gmacs(model, cfg.img_size)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] không đếm được GMAC: {e}")
        gmacs = float("nan")
    model.to(device)
    criterion, cw = build_loss_for(cfg, train_df)
    criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
    val_criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) if cfg.max_train_batches is None \
        else min(len(train_loader), cfg.max_train_batches)
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch)
    scaler = torch.amp.GradScaler(device.type) if (cfg.amp and amp_dtype(device) == torch.float16) else None
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    mix_rng = np.random.default_rng(cfg.seed + 12345)

    meta = {"config": dataclasses.asdict(cfg), "weight_tag": weight_tag(model), "device": str(device),
            "amp_dtype": str(amp_dtype(device)) if cfg.amp else "fp32", "params_m": params_m,
            "trainable_params_m": trainable_m, "gmacs": gmacs, "class_weights": cw,
            "split_check": split_info, "versions": lib_versions(),
            "param_groups": [{"name": g["name"], "n": sum(p.numel() for p in g["params"]), "lr": g["lr"],
                              "weight_decay": g["weight_decay"]} for g in optimizer.param_groups]}
    (rd / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"[{cfg.exp_id} seed{cfg.seed}] {meta['weight_tag']} params={params_m:.2f}M GMAC={gmacs:.2f} "
          f"device={device} amp={meta['amp_dtype']}")

    # 5. vòng epoch
    history, all_lrs = [], []
    best = {"f1": -1.0, "epoch": -1}
    start_epoch, elapsed_before = 1, 0.0
    last_path = rd / "last.pt"
    if last_path.exists():  # tiếp tục lần chạy bị ngắt (checkpoint cuối mỗi epoch)
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler is not None and ck.get("scaler") is not None:
            scaler.load_state_dict(ck["scaler"])
        if ema is not None:
            ema.module.load_state_dict(ck["ema"])
            ema.updates = ck["ema_updates"]
        history, all_lrs, best = ck["history"], ck["lrs"], ck["best"]
        mix_rng.bit_generator.state = ck["mix_rng"]
        torch.set_rng_state(ck["torch_rng"].cpu())
        start_epoch, elapsed_before = ck["epoch"] + 1, ck["elapsed"]
        print(f"[{cfg.exp_id} seed{cfg.seed}] RESUME từ epoch {start_epoch} (last.pt)", flush=True)
    t_start = time.perf_counter() - elapsed_before
    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device,
                             ema, mix_rng)
        if device.type == "mps":
            torch.mps.synchronize()
        t_train = time.perf_counter() - t0
        eval_model = ema.module if ema is not None else model
        names, y_val, logits_val, val_loss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
        probs = softmax_np(logits_val)
        m = compute_metrics(y_val, probs.argmax(1), probs)
        all_lrs.extend(tr["lrs"])
        h = {"epoch": epoch, "train_loss": tr["train_loss"], "train_acc": tr["train_acc"], "val_loss": val_loss,
             "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_bal_acc": m["balanced_acc"],
             "val_ece": m["ece"], "lr_end": tr["lrs"][-1] if tr["lrs"] else float("nan"),
             "train_time_s": t_train, "epoch_time_s": time.perf_counter() - t0}
        history.append(h)
        improved = m["macro_f1"] > best["f1"]  # hòa -> giữ epoch sớm hơn
        if improved:
            best = {"f1": m["macro_f1"], "epoch": epoch}
            torch.save(eval_model.state_dict(), rd / "best.pt")
            np.save(rd / "val_logits.npy", logits_val)
            (rd / "val_names.json").write_text(json.dumps(names))
        pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                    "ema": ema.module.state_dict() if ema else None, "ema_updates": ema.updates if ema else 0,
                    "history": history, "lrs": all_lrs, "best": best, "epoch": epoch,
                    "mix_rng": mix_rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "elapsed": time.perf_counter() - t_start}, last_path)
        print(f"[{cfg.exp_id} seed{cfg.seed}] ep{epoch:02d} train_loss={h['train_loss']:.4f} "
              f"val_loss={val_loss:.4f} val_F1={m['macro_f1']:.4f} val_top1={m['top1']:.4f} "
              f"t={h['epoch_time_s']:.0f}s{' *' if improved else ''}", flush=True)
    total_time = time.perf_counter() - t_start

    last_path.unlink(missing_ok=True)

    # 6. nạp checkpoint tốt nhất, lưu dự đoán val
    eval_model = ema.module if ema is not None else model
    eval_model.load_state_dict(torch.load(rd / "best.pt", map_location=device))
    names, y_val, logits_val, val_loss = evaluate(eval_model, val_loader, val_criterion, device, cfg.amp)
    np.save(rd / "val_logits.npy", logits_val)
    np.save(rd / "val_labels.npy", y_val)
    probs_val = softmax_np(logits_val)
    save_predictions(pred_path(cfg, "val"), names, y_val, probs_val)
    m_val = compute_metrics(y_val, probs_val.argmax(1), probs_val)

    # 7. test: chỉ ở Bước 4, đúng MỘT lần
    m_test = None
    if cfg.save_test_predictions:
        test_loader = D.make_loader(test_df, cfg.images_dir, tf_eval, 128, False, None, cfg.num_workers)
        tn, y_te, logits_te, _ = evaluate(eval_model, test_loader, val_criterion, device, cfg.amp)
        np.save(rd / "test_logits.npy", logits_te)
        p_te = softmax_np(logits_te)
        save_predictions(pred_path(cfg, "test"), tn, y_te, p_te)
        m_test = compute_metrics(y_te, p_te.argmax(1), p_te)

    # 8. history, curves, độ trễ sơ bộ, tóm tắt
    lat = None
    if cfg.measure_latency:
        from benchmark import latency_report
        lat = latency_report(eval_model, 1, cfg.img_size, "fp32", device.type, warmup=10, iters=50)
    plot_curves(history, curve_path(cfg), f"{cfg.exp_id} · {cfg.backbone} · seed {cfg.seed}"
                + (f" · {cfg.desc}" if cfg.desc else ""), all_lrs)
    pd.DataFrame({"step": np.arange(1, len(all_lrs) + 1), "lr": all_lrs}).to_csv(rd / "lr.csv", index=False)
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": meta["weight_tag"],
        "best_epoch": best["epoch"], "val_macro_f1": m_val["macro_f1"], "val_top1": m_val["top1"],
        "val_bal_acc": m_val["balanced_acc"], "val_ece": m_val["ece"],
        "val_f1_per_class": m_val["f1"].tolist(), "val_recall_per_class": m_val["recall"].tolist(),
        "train_time_per_epoch_s": float(np.mean([h["train_time_s"] for h in history])),
        "total_time_s": total_time, "params_m": params_m, "trainable_params_m": trainable_m, "gmacs": gmacs,
        "latency_b1_fp32": lat, "device": str(device),
        "test_macro_f1": None if m_test is None else m_test["macro_f1"],
    }
    (rd / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(f"[{cfg.exp_id} seed{cfg.seed}] DONE best_ep={best['epoch']} val_F1={m_val['macro_f1']:.4f} "
          f"val_top1={m_val['top1']:.4f} {total_time / 60:.1f} min", flush=True)
    return summary


def _coerce(value: str, field: dataclasses.Field):
    v = value.strip()
    if v.lower() in ("none", "null"):
        return None
    t = field.type if isinstance(field.type, str) else getattr(field.type, "__name__", str(field.type))
    if t.startswith("bool"):
        if v.lower() in ("1", "true", "yes", "on"):
            return True
        if v.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{field.name}: '{value}' không phải bool")
    if t.startswith("int"):
        return int(v)
    if t.startswith("float"):
        return float(v)
    return v


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict, ép kiểu theo field của Config."""
    fields = {f.name: f for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"'{pair}' không có dạng KEY=VALUE")
        k, v = pair.split("=", 1)
        k = k.strip()
        if k not in fields:
            raise KeyError(f"'{k}' không phải field của Config; các field: {sorted(fields)}")
        out[k] = _coerce(v, fields[k])
    return out


def main() -> None:
    """`python train.py --set exp_id=B01 backbone=resnet50 seed=0`."""
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
