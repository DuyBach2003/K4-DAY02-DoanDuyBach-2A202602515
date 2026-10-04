"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Quy tắc đo:
  - warmup: bỏ >= 10 lần chạy đầu
  - đồng bộ GPU TRƯỚC và SAU đoạn cần đo: torch.cuda.synchronize() trên CUDA, torch.mps.synchronize()
    trên Apple GPU (MPS) - MPS cũng chạy bất đồng bộ như CUDA
  - >= 50 lần đo, báo cáo p50, p95, p99 (không chỉ trung bình)
  - ghi rõ thiết bị, dtype, batch, độ phân giải, có/không gộp BN, phiên bản torch
  - KHÔNG tính tiền xử lý (đọc/giải mã JPEG, resize): chỉ đo forward của model trên tensor đã ở trên thiết bị.
"""
from __future__ import annotations

import platform
import subprocess
import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian `fn()` (mili-giây): warmup lần đầu bỏ đi; mỗi lần đo sync() -> t0 -> fn() -> sync() -> t1."""
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99)), "mean": float(a.mean()), "std": float(a.std(ddof=1)),
            "n": iters, "warmup": warmup}


def device_name(device: str) -> str:
    import torch

    if device == "cuda":
        return torch.cuda.get_device_name(0)
    if device == "mps":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        except OSError:
            chip = platform.machine()
        return f"{chip} GPU (MPS)"
    return f"CPU {platform.processor() or platform.machine()}"


def sync_fn(device: str):
    import torch

    if device == "cuda":
        return torch.cuda.synchronize
    if device == "mps":
        return torch.mps.synchronize
    return None


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, forward=None, fused_bn: bool = False) -> dict:
    """Độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size).

    dtype: "fp32" | "amp" (autocast fp16 trên CUDA/MPS) | "fp16" (model.half() + input half).
    `forward(model, x)` tuỳ chọn thay cho model(x) (dùng cho TTA K view).
    """
    import copy

    import torch

    dev = torch.device(device)
    m = model
    if dtype == "fp16":
        m = copy.deepcopy(model).half()
    m = m.to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        x = x.half()
    fwd = forward if forward is not None else (lambda mm, xx: mm(xx))

    def fn():
        with torch.inference_mode():
            if dtype == "amp":
                with torch.autocast(device_type=dev.type, dtype=torch.float16):
                    fwd(m, x)
            else:
                fwd(m, x)

    r = bench(fn, warmup, iters, sync_fn(device))
    return {"gpu": device_name(device), "dtype": dtype, "batch": batch_size, "img_size": img_size,
            "fused_bn": fused_bn, "p50": r["p50"], "p95": r["p95"], "p99": r["p99"], "mean": r["mean"],
            "n": r["n"], "warmup": r["warmup"], "images_per_s": batch_size / (r["p50"] / 1000.0),
            "torch": torch.__version__, "preprocessing_included": False}


def tta_latency(model, k_views: int, view_fn=None, **kw) -> dict:
    """Độ trễ TTA K view: đo thật forward K lượt (view_fn(x) -> list K batch), so sánh với K * p50 một lượt."""
    import torch

    if view_fn is None:
        def view_fn(x):
            return [x if i % 2 == 0 else torch.flip(x, dims=[3]) for i in range(k_views)]

    def forward(m, x):
        return [m(v) for v in view_fn(x)]

    one = latency_report(model, **kw)
    many = latency_report(model, forward=forward, **kw)
    many["k_views"] = k_views
    many["k_times_single_p50"] = k_views * one["p50"]
    many["ratio_vs_single_p50"] = many["p50"] / one["p50"]
    return many
