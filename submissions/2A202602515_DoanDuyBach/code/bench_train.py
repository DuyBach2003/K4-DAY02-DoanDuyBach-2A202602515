"""Đo tốc độ huấn luyện của các backbone trong CÙNG điều kiện (GPU rảnh, không đọc dữ liệu):
batch 64, 224x224, AdamW, AMP fp16 + GradScaler như train.py; warmup 5 bước, đo 20 bước có synchronize.

Lý do: thời gian/epoch ghi trong log Bước 1 bị nhiễu (B05 chạy khi tiến trình còn trong sandbox chậm hơn,
B03 chạy cùng lúc với một job CPU) nên không so sánh công bằng được.

    python code/bench_train.py     (từ thư mục bài nộp) -> logs/bench_train.json
"""
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn.functional as F

BACKBONES = {"B01": "resnet50", "B02": "convnext_tiny", "B03": "deit_small_patch16_224",
             "B04": "swin_tiny_patch4_window7_224", "B05": "efficientnet_b0", "B06": "mobilenetv3_large_100"}


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    sync = torch.mps.synchronize if dev.type == "mps" else (torch.cuda.synchronize if dev.type == "cuda" else (lambda: None))
    out = {}
    for exp, name in BACKBONES.items():
        m = timm.create_model(name, pretrained=False, num_classes=9).to(dev).train()
        opt = torch.optim.AdamW(m.parameters(), 1e-4)
        scaler = torch.amp.GradScaler(dev.type) if dev.type != "cpu" else None
        x, y = torch.randn(64, 3, 224, 224, device=dev), torch.randint(0, 9, (64,), device=dev)

        def step():
            with torch.autocast(dev.type, dtype=torch.float16, enabled=scaler is not None):
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
        out[exp] = {"backbone": name, "train_img_per_s": 64 / dt, "est_epoch_s_10501_imgs": 10501 / (64 / dt)}
        print(exp, json.dumps(out[exp]), flush=True)
        del m, opt
        if dev.type == "mps":
            torch.mps.empty_cache()
    Path("logs/bench_train.json").write_text(json.dumps({"device": str(dev), "torch": torch.__version__,
                                                         "results": out}, indent=2))


if __name__ == "__main__":
    main()
