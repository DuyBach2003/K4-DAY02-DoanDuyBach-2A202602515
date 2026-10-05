"""Đo tốc độ train ResNet-50 theo dtype trên máy này, làm căn cứ chọn dtype cho AMP (report mục 2.4): batch 64, 224 × 224,
AdamW; FP32, autocast FP16 + GradScaler (như train.py), autocast BF16; warmup 5 bước, đo 20 bước có synchronize.
Không đọc dữ liệu (đầu vào ngẫu nhiên), chạy khi máy rảnh.

    python code/bench_amp_dtype.py   (từ thư mục bài nộp) -> logs/bench_amp_dtype.json
"""
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.mps.synchronize if dev.type == "mps" else (torch.cuda.synchronize if dev.type == "cuda" else (lambda: None))
    out = {}
    for name, dtype in (("fp32", None), ("fp16", torch.float16), ("bf16", torch.bfloat16)):
        torch.manual_seed(0)
        m = timm.create_model("resnet50", pretrained=False, num_classes=9).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), 1e-4)
        scaler = torch.amp.GradScaler(dev.type) if dtype == torch.float16 else None
        x, y = torch.randn(64, 3, 224, 224, device=dev), torch.randint(0, 9, (64,), device=dev)

        def step():
            with torch.autocast(dev.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                loss = F.cross_entropy(m(x).float(), y)
            opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

        for _ in range(5):
            step()
        sync(); t = time.perf_counter()
        for _ in range(20):
            step()
        sync(); dt = (time.perf_counter() - t) / 20
        out[name] = {"train_img_per_s": 64 / dt, "step_ms": dt * 1000}
        print(name, json.dumps(out[name]), flush=True)
        del m, opt
        if dev.type == "mps":
            torch.mps.empty_cache()
    Path("logs/bench_amp_dtype.json").write_text(json.dumps({"device": str(dev), "torch": torch.__version__,
                                                             "model": "resnet50", "batch": 64, "img_size": 224,
                                                             "results": out}, indent=2))


if __name__ == "__main__":
    main()
