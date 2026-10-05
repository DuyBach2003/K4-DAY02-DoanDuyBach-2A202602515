"""Điểm thưởng (RUBRIC mục 2), CHỈ dùng VAL của fold 0 (không đụng test):

  shift: lệch phân phối tự tạo trên ảnh val (thiếu sáng, mờ, nhiễu; 2 mức mỗi loại). Đo macro-F1, top-1 và ECE trước/sau
         temperature scaling của chung kết F01 (3 seed, ảnh đầy đủ 288) và mốc T00 (3 seed, CenterCrop 224).
         T "sạch" khớp trên val sạch, đúng như chung kết; T "oracle" khớp lại trên chính dữ liệu lệch (có dùng nhãn), chỉ để
         biết temperature scaling còn sửa được tới đâu, không dùng để chọn gì.
  bn:    test-time adaptation bằng chuẩn hoá lại thống kê BatchNorm trên chính ảnh lệch (không nhãn) cho hai CNN có BN
         (B05, B06), so với thống kê gốc và precise-BN (thống kê từ ảnh train sạch).
  tent:  Tent (Wang và cộng sự, ICLR 2021) cho F01 seed 0 (DeiT-S chỉ có LayerNorm): cập nhật γ/β của mọi LayerNorm bằng
         cực tiểu entropy dự đoán, online một lượt trên ảnh lệch (không nhãn). Siêu tham số cố định theo bài báo Tent
         cho ImageNet (SGD, momentum 0,9, lr 0,00025, batch 64), không tinh chỉnh.

Biến dạng áp lên ảnh gốc 256 × 256 (8 bit) TRƯỚC khi resize/crop như lúc suy luận; nhiễu có seed cố định theo từng ảnh.

    python code/bonus_shift_tta.py shift|bn|tent   (từ thư mục bài nộp) -> logs/bonus_shift.json, logs/bonus_tta.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402
import dataset as D  # noqa: E402
from eval import compute_metrics  # noqa: E402
from inference import apply_temperature, fit_temperature, recalibrate_bn  # noqa: E402
from step3_inference import load_model  # noqa: E402

BACKBONE = "deit_small_patch16_224"
SEEDS = (0, 1, 2)
# tên -> (loại, mức); "clean" là ảnh gốc
CONDITIONS = {"clean": (None, None), "dark_0.5": ("dark", 0.5), "dark_0.3": ("dark", 0.3),
              "blur_1.5": ("blur", 1.5), "blur_3": ("blur", 3.0), "noise_0.05": ("noise", 0.05),
              "noise_0.10": ("noise", 0.10)}
# cấu hình suy luận của chung kết và mốc (Bước 4)
GROUPS = {"F01": (288, False), "T00": (224, True)}
IMAGES, LABELS = "../../data/images", "../../data/labels"


def corrupt(img, kind, level, seed):
    """Ảnh PIL RGB -> ảnh PIL RGB đã biến dạng (vẫn 8 bit như ảnh chụp thật)."""
    from PIL import Image, ImageFilter

    if kind is None:
        return img
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(radius=level))
    a = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "dark":
        a = a * level
    elif kind == "noise":
        a = a + np.random.default_rng(seed).normal(0.0, level, a.shape).astype(np.float32)
    else:
        raise ValueError(kind)
    return Image.fromarray((np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8))


class ShiftDataset(D.DeepWeedsDataset):
    def __init__(self, df, transform, cond):
        super().__init__(df, IMAGES, transform)
        self.kind, self.level = CONDITIONS[cond]

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.images_dir / self.filenames[i]) as im:
            img = corrupt(im.convert("RGB"), self.kind, self.level, seed=i)
        return self.transform(img), self.labels[i], self.filenames[i]


def eval_transform(size, crop):
    """Giống hệt step3_inference.loader_for: crop -> Resize(size/0.875) + CenterCrop; full -> resize cả ảnh."""
    from torchvision import transforms as T

    if crop:
        return D.build_transforms(False, size)
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)])


def loader(df, size, crop, cond, workers=3):
    from torch.utils.data import DataLoader

    return DataLoader(ShiftDataset(df, eval_transform(size, crop), cond), batch_size=64, shuffle=False,
                      num_workers=workers)


def logits_of(models, dl, device):
    """Một lượt dữ liệu, chạy mọi model trên cùng batch: -> (y, [logits của từng model])."""
    import torch

    ys, outs = [], [[] for _ in models]
    with torch.inference_mode():
        for x, y, _ in dl:
            x = x.to(device)
            for k, m in enumerate(models):
                outs[k].append(m(x).float().cpu())
            ys.append(y)
    return torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def metrics(y, z, T=1.0):
    p = apply_temperature(z, T)
    r = compute_metrics(y, p.argmax(1), p)
    return {"macro_f1": r["macro_f1"], "top1": r["top1"], "ece": r["ece"], "conf": float(p.max(1).mean()),
            "f1_chinee": float(r["f1"][0]), "f1_snake": float(r["f1"][7])}


def mean_std(v):
    return {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else float("nan")}


def part_shift(device, va):
    out = {"split": "val fold 0", "n_images": len(va), "conditions": CONDITIONS, "groups": {}, "results": {}}
    for exp, (size, crop) in GROUPS.items():
        out["groups"][exp] = {"img_size": size, "eval": "CenterCrop" if crop else "ảnh đầy đủ"}
        models = [load_model(exp, BACKBONE, s, device) for s in SEEDS]
        t_clean = None
        res = {}
        for cond in CONDITIONS:
            y, zs = logits_of(models, loader(va, size, crop, cond), device)
            if cond == "clean":
                t_clean = [fit_temperature(z, y) for z in zs]
            per_seed = []
            for s, z, tc in zip(SEEDS, zs, t_clean):
                t_or = fit_temperature(z, y)
                m1, mc, mo = metrics(y, z, 1.0), metrics(y, z, tc), metrics(y, z, t_or)
                per_seed.append({"seed": s, "T_clean": tc, "T_oracle": t_or, "macro_f1": m1["macro_f1"],
                                 "top1": m1["top1"], "f1_chinee": m1["f1_chinee"], "f1_snake": m1["f1_snake"],
                                 "ece_T1": m1["ece"], "ece_Tclean": mc["ece"], "ece_Toracle": mo["ece"],
                                 "conf_T1": m1["conf"], "conf_Tclean": mc["conf"]})
            summ = {k: mean_std([r[k] for r in per_seed]) for k in per_seed[0] if k != "seed"}
            res[cond] = {"per_seed": per_seed, "summary": summ}
            print(exp, cond, {k: round(v["mean"], 4) for k, v in summ.items()}, flush=True)
        out["results"][exp] = res
        del models
    Path("logs/bonus_shift.json").write_text(json.dumps(out, indent=2))


def load_tta():
    p = Path("logs/bonus_tta.json")
    return json.loads(p.read_text()) if p.exists() else {}


def part_bn(device, tr, va):
    out = load_tta()
    out["bn"] = {"note": "CNN có BN, seed 0, CenterCrop 224; thống kê BN: gốc (lúc train), precise-BN (ảnh train sạch), "
                         "thích ứng (chính ảnh lệch, không nhãn)", "results": {}}
    for exp, bb in (("B05", "efficientnet_b0"), ("B06", "mobilenetv3_large_100")):
        src = load_model(exp, bb, 0, device)
        pbn = recalibrate_bn(src, loader(tr, 224, True, "clean"), device)
        res = {}
        for cond in CONDITIONS:
            dl = loader(va, 224, True, cond)
            adapted = recalibrate_bn(src, dl, device)
            y, (z0, z1, z2) = logits_of([src, pbn, adapted], dl, device)
            res[cond] = {"goc": metrics(y, z0), "precise_bn": metrics(y, z1), "bn_adapt": metrics(y, z2)}
            print(exp, cond, {k: round(v["macro_f1"], 4) for k, v in res[cond].items()}, flush=True)
            del adapted
        out["bn"]["results"][exp] = res
    Path("logs/bonus_tta.json").write_text(json.dumps(out, indent=2))


def part_tent(device, va, lr=0.00025, momentum=0.9):
    import torch
    import torch.nn as nn

    out = load_tta()
    size, crop = GROUPS["F01"]
    src = load_model("F01", BACKBONE, 0, device)
    y0, (z0,) = logits_of([src], loader(va, size, crop, "clean"), device)
    t_clean = fit_temperature(z0, y0)
    out["tent"] = {"note": "F01 seed 0 (DeiT-S), ảnh đầy đủ 288; Tent online 1 lượt, chỉ cập nhật γ/β của LayerNorm",
                   "optimizer": {"name": "SGD", "lr": lr, "momentum": momentum, "batch": 64},
                   "T_clean": t_clean, "results": {}}
    for cond in CONDITIONS:
        if cond == "clean":
            continue
        dl = loader(va, size, crop, cond)
        y, (z_src,) = logits_of([src], dl, device)
        m = copy.deepcopy(src).eval()
        params = []
        for mod in m.modules():
            for p in mod.parameters(recurse=False):
                p.requires_grad_(isinstance(mod, nn.LayerNorm))
                if isinstance(mod, nn.LayerNorm):
                    params.append(p)
        opt = torch.optim.SGD(params, lr=lr, momentum=momentum)
        zs = []
        for x, _, _ in dl:
            logits = m(x.to(device))
            p = logits.softmax(1)
            loss = -(p * logits.log_softmax(1)).sum(1).mean()  # entropy dự đoán, không dùng nhãn
            opt.zero_grad()
            loss.backward()
            opt.step()
            zs.append(logits.detach().float().cpu())  # dự đoán online (trước bước cập nhật trên batch đó), như Tent
        z_tent = torch.cat(zs).numpy()
        out["tent"]["results"][cond] = {"n_params": int(sum(p.numel() for p in params)),
                                        "truoc": metrics(y, z_src), "truoc_Tclean": metrics(y, z_src, t_clean),
                                        "tent": metrics(y, z_tent), "tent_Tclean": metrics(y, z_tent, t_clean)}
        r = out["tent"]["results"][cond]
        print("tent", cond, round(r["truoc"]["macro_f1"], 4), "->", round(r["tent"]["macro_f1"], 4), flush=True)
        del m, opt
    Path("logs/bonus_tta.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["shift", "bn", "tent"])
    args = ap.parse_args()
    device = TR.get_device()
    tr, va, _ = D.load_split(LABELS, 0)
    if args.part == "shift":
        part_shift(device, va)
    elif args.part == "bn":
        part_bn(device, tr, va)
    else:
        part_tent(device, va)


if __name__ == "__main__":
    main()
