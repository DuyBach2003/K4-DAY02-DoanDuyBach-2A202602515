"""Danh sách thí nghiệm và runner chạy tuần tự. Mọi thí nghiệm đi qua MỘT hàm train.run(Config(...)).

Chạy từ thư mục bài nộp:
    python code/experiments.py B01 B02 ...          # chạy các exp_id (seed 0)
    python code/experiments.py F01 --seeds 0 1 2    # nhiều seed
Bỏ qua lần chạy đã xong (có runs/<exp>/seed<k>/summary.json). Log đầy đủ ở logs/<exp>_seed<k>.log;
config.json, history.csv, summary.json được chép sang logs/<exp>/seed<k>/ để nộp (runs/ chứa checkpoint, không commit).

Giảm bớt do ngân sách GPU (GUIDE mục 7), áp dụng cho MỌI thí nghiệm: 10 epoch (thay vì 12-15).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import shutil
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

COMMON = dict(epochs=10, num_workers=6, images_dir="../../data/images", labels_dir="../../data/labels",
              out_dir="runs", pred_dir="predictions", curves_dir="curves")

# ---- Bước 1: backbone, cùng công thức nền T00 (GUIDE mục 1.4), seed 0 ----
BACKBONES = {
    "B01": dict(backbone="resnet50", desc="resnet50"),
    "B02": dict(backbone="convnext_tiny", desc="convnext_tiny"),
    "B03": dict(backbone="deit_small_patch16_224", desc="deit_small"),
    "B04": dict(backbone="swin_tiny_patch4_window7_224", desc="swin_tiny"),
    "B05": dict(backbone="efficientnet_b0", desc="efficientnet_b0"),
    "B06": dict(backbone="mobilenetv3_large_100", desc="mobilenetv3_large"),
}

# ---- Chẩn đoán: CNN có BN underfit ở Bước 1; loại trừ ảnh hưởng của AMP fp16 trên MPS (so với B06) ----
DIAG = {
    "D01": dict(backbone="mobilenetv3_large_100", amp=False, desc="mobilenetv3_fp32"),
}

# ---- Bước 2: công thức huấn luyện trên backbone đã chọn ở Bước 1 ----
# T00 = công thức nền trên backbone đó; seed 0 của T00 CHÍNH LÀ lần chạy Bước 1 tương ứng (cùng cấu hình),
# được sao chép bằng `python code/experiments.py --alias <Bxx> T00` thay vì chạy lại.
STEP2_BACKBONE = "deit_small_patch16_224"
STEP2_ALIAS = "B03"
_b = dict(backbone=STEP2_BACKBONE)
TRAINING = {
    "T00": dict(_b, desc="baseline"),
    "T01": dict(_b, init="frozen", desc="frozen"),
    "T02": dict(_b, aug="mildcrop", desc="mildcrop"),
    "T03": dict(_b, mix="cutmix", mix_alpha=1.0, desc="cutmix"),
    "T04": dict(_b, aug="trivial", desc="trivialaug"),
    "T05": dict(_b, loss="ls", label_smoothing=0.1, desc="labelsmooth"),
    "T06": dict(_b, loss="focal", focal_gamma=2.0, desc="focal"),
    "T07": dict(_b, loss="ce_weighted", desc="ce_weighted"),
    "T08": dict(_b, ema_decay=0.999, desc="ema"),
    # kết hợp các yếu tố có Δ > 0 ở các trục đơn lẻ: label smoothing (T05) + crop nhẹ (T02)
    "T09": dict(_b, loss="ls", label_smoothing=0.1, aug="mildcrop", desc="ls_mildcrop"),
}
TRAINING_META = {
    "T00": {"axis": "-", "diff": "công thức nền (= B03, seed 0 dùng lại lần chạy B03)"},
    "T01": {"axis": "A. Khởi tạo", "diff": "đóng băng backbone, chỉ train head (linear probe)"},
    "T02": {"axis": "B. Augmentation", "diff": "RandomResizedCrop scale (0.35, 1) thay vì (0.08, 1)"},
    "T03": {"axis": "B. Augmentation", "diff": "+ CutMix (alpha = 1.0, xác suất 1)"},
    "T04": {"axis": "B. Augmentation", "diff": "+ TrivialAugmentWide"},
    "T05": {"axis": "C. Loss", "diff": "label smoothing eps = 0.1"},
    "T06": {"axis": "C. Loss", "diff": "focal loss gamma = 2"},
    "T07": {"axis": "C. Loss", "diff": "CE có trọng số lớp 1/n_c (chuẩn hoá trung bình 1)"},
    "T08": {"axis": "F. Chính quy hoá", "diff": "EMA trọng số, decay 0.999 (đánh giá bằng trọng số EMA)"},
    "T09": {"axis": "Kết hợp (B + C)", "diff": "label smoothing 0.1 (T05) + RandomResizedCrop scale (0.35, 1) (T02)"},
}

# ---- Bước 4: chung kết (điền sau Bước 2-3) ----
# F01 = công thức T09 (seed 0 dùng lại lần chạy T09, sao chép bằng --alias T09 F01), seed 1, 2 chạy mới.
FINAL = {"F01": dict(TRAINING["T09"], desc="final_ls_mildcrop")}
FINAL_META: dict[str, str] = {
    "T00": f"mốc: {STEP2_BACKBONE} + công thức nền T00 + suy luận 1 view I00",
    "F01": f"chung kết: {STEP2_BACKBONE} + T09 (label smoothing 0.1 + crop nhẹ) + suy luận ảnh đầy đủ 288, 1 view "
           "(I04_288) + temperature scaling (T khớp trên val từng seed)",
}

# ---- Điểm thưởng (RUBRIC mục 2): cấu hình chung kết trên fold khác, seed 0, dùng đủ bộ ba file của fold đó (S6) ----
FOLDS = {f"F01_fold{k}": dict(FINAL["F01"], fold=k) for k in (1, 2)}

REGISTRY = {**BACKBONES, **DIAG, **TRAINING, **FINAL, **FOLDS}


def make_config(exp_id: str, seed: int, **extra):
    from train import Config

    kw = {**COMMON, **REGISTRY[exp_id], **extra}
    return Config(exp_id=exp_id, seed=seed, **kw)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            if not st.closed:  # logger của thư viện có thể giữ Tee của lần chạy trước (file log đã đóng)
                st.write(s)
                st.flush()

    def flush(self):
        for st in self.streams:
            if not st.closed:
                st.flush()

    @property
    def closed(self):
        return False


def run_one(exp_id: str, seed: int, **extra) -> dict | None:
    import train

    cfg = make_config(exp_id, seed, **extra)
    rd = train.run_dir(cfg)
    if (rd / "summary.json").exists():
        print(f"skip {exp_id} seed{seed} (đã có summary.json)")
        return None
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / f"{exp_id}_seed{seed}.log", "a") as f, \
            contextlib.redirect_stdout(Tee(sys.__stdout__, f)), contextlib.redirect_stderr(Tee(sys.__stderr__, f)):
        print(f"=== {exp_id} seed{seed}: {dataclasses.asdict(cfg)}")
        try:
            summary = train.run(cfg)
        except Exception:
            traceback.print_exc()
            return None
    dst = log_dir / exp_id / f"seed{seed}"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("config.json", "history.csv", "summary.json", "lr.csv"):
        shutil.copy(rd / name, dst / name)
    return summary


def alias_run(src: str, dst: str, seed: int = 0) -> None:
    """Dùng lại lần chạy `src` (cùng cấu hình) làm `dst`: chép runs/, logs/, predictions val và ảnh biểu đồ."""
    import train

    s_cfg, d_cfg = make_config(src, seed), make_config(dst, seed)
    a, b = dataclasses.asdict(s_cfg), dataclasses.asdict(d_cfg)
    diff = {k for k in a if a[k] != b[k]} - {"exp_id", "desc"}
    if diff:
        raise ValueError(f"{src} và {dst} khác cấu hình ở {diff}; không được dùng lại")
    if not (train.run_dir(s_cfg) / "summary.json").exists():
        raise FileNotFoundError(f"{src} seed{seed} chưa chạy xong (không có summary.json); không sao chép")
    shutil.copytree(train.run_dir(s_cfg), train.run_dir(d_cfg), dirs_exist_ok=True)
    shutil.copytree(Path("logs") / src / f"seed{seed}", Path("logs") / dst / f"seed{seed}", dirs_exist_ok=True)
    shutil.copy(train.pred_path(s_cfg, "val"), train.pred_path(d_cfg, "val"))
    shutil.copy(train.curve_path(s_cfg), train.curve_path(d_cfg))
    note = Path("logs") / dst / f"seed{seed}" / "ALIAS.txt"
    note.write_text(f"{dst} seed{seed} là bản sao của {src} seed{seed} (cấu hình giống hệt, không chạy lại).\n")
    print(note.read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_ids", nargs="*")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--alias", nargs=2, metavar=("SRC", "DST"))
    args = ap.parse_args()
    if args.alias:
        alias_run(*args.alias)
        return
    for exp_id in args.exp_ids:
        if exp_id not in REGISTRY:
            raise KeyError(f"{exp_id} chưa có trong REGISTRY")
    for exp_id in args.exp_ids:
        for seed in args.seeds:
            run_one(exp_id, seed)


if __name__ == "__main__":
    main()
