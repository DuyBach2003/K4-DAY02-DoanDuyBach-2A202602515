"""Chẩn đoán: BatchNorm ở train mode trên MPS (fp32 và autocast fp16) có cho cùng output/gradient như CPU không?

Bối cảnh: ở Bước 1 mọi CNN có BatchNorm (ResNet-50, EfficientNet-B0, MobileNetV3) underfit nặng (train loss 0.25-0.42
sau 10 epoch) trong khi DeiT-S (LayerNorm) đạt train loss 0.06. Script này loại trừ (hoặc xác nhận) lỗi số học của MPS.

    python code/diag_bn_mps.py      (từ thư mục bài nộp) -> logs/diag_bn_mps.json
"""
import copy
import json
from pathlib import Path

import timm
import torch
import torch.nn.functional as F


def one_step(m, x, y, dev, amp):
    mm = copy.deepcopy(m).to(dev).train()
    with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.float16, enabled=amp):
        out = mm(x.to(dev))
    loss = F.cross_entropy(out.float(), y.to(dev))
    loss.backward()
    bn = [mod for mod in mm.modules() if isinstance(mod, torch.nn.BatchNorm2d)][5]
    g = torch.cat([p.grad.flatten().float().cpu() for p in mm.parameters() if p.grad is not None])
    return out.float().cpu().detach(), g, bn.running_mean.cpu().clone(), loss.item()


def main():
    torch.manual_seed(0)
    results = {}
    for name in ["resnet50", "efficientnet_b0"]:
        m = timm.create_model(name, pretrained=True, num_classes=9)
        x, y = torch.randn(16, 3, 224, 224), torch.randint(0, 9, (16,))
        ref = one_step(m, x, y, "cpu", False)
        for tag, amp in (("mps_fp32", False), ("mps_fp16", True)):
            o, g, rm, l = one_step(m, x, y, "mps", amp)
            r = {"loss": l, "loss_cpu": ref[3], "out_maxdiff": float((o - ref[0]).abs().max()),
                 "grad_cosine": float(F.cosine_similarity(g, ref[1], dim=0)),
                 "grad_norm_ratio": float(g.norm() / ref[1].norm()),
                 "bn_running_mean_maxdiff": float((rm - ref[2]).abs().max())}
            results[f"{name}/{tag}"] = r
            print(name, tag, json.dumps({k: round(v, 5) for k, v in r.items()}), flush=True)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/diag_bn_mps.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
