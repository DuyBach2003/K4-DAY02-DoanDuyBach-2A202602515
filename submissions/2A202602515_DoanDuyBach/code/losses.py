"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Liên hệ slide Day 2: label smoothing (trang 56), focal loss (trang 57), Mixup/CutMix (trang 48).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    """Trả về hàm loss theo `kind`: "ce", "ls" (label smoothing), "focal", "ce_weighted".

    kw: smoothing (ls), gamma và alpha (focal), weight (ce_weighted, tensor độ dài 9).
    """
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), kw.get("alpha"))
    if kind == "ce_weighted":
        w = kw.get("weight")
        if w is None:
            raise ValueError("ce_weighted cần weight")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(w, dtype=torch.float32))
    raise ValueError(f"loss không hợp lệ: {kind}")


class LabelSmoothingCE(nn.Module):
    """Cross-entropy với label smoothing: q'(k) = (1 - eps) * 1[k == y] + eps / K  (slide trang 56).

    Tự cài đặt (không dùng label_smoothing của PyTorch) để thấy rõ công thức; eps = 0 cho đúng CE.
    """

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing phải trong [0, 1)")
        self.smoothing = smoothing

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        k = logits.shape[1]
        q = torch.full_like(logp, self.smoothing / k)
        q.scatter_(1, target.view(-1, 1), 1.0 - self.smoothing + self.smoothing / k)
        return -(q * logp).sum(1).mean()


class FocalLoss(nn.Module):
    """Focal loss nhiều lớp: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (slide trang 57).

    alpha: None hoặc vector trọng số theo lớp (độ dài K). Trung bình theo batch. gamma = 0, alpha = None ≡ CE.
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=1)
        logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp(min=0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số theo lớp từ số ảnh mỗi lớp trong tập TRAIN.

    - beta = 0: w_c ∝ 1 / n_c, chuẩn hoá về trung bình 1
    - beta > 0: class-balanced, w_c = (1 - beta) / (1 - beta ** n_c) (Cui et al. 2019), chuẩn hoá tổng = số lớp
    """
    n = np.asarray(counts, dtype=np.float64)
    if (n <= 0).any():
        raise ValueError("mọi lớp phải có ít nhất 1 ảnh")
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
    else:
        w = 1.0 / n
    w = w / w.sum() * len(n)  # tổng = K, tức trung bình 1
    return torch.tensor(w, dtype=torch.float32)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix", generator: np.random.Generator | None = None):
    """Trộn một batch ảnh và nhãn. Trả về (x_mix, (y_a, y_b, lam)) với y_a = y, y_b = y[perm].

    - mixup : x_mix = lam * x + (1 - lam) * x[perm]
    - cutmix: dán một hộp của x[perm] vào x; lam được tính lại theo DIỆN TÍCH THỰC của hộp sau khi cắt biên.
    """
    rng = generator if generator is not None else np.random.default_rng()
    lam = float(rng.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.from_numpy(rng.permutation(x.shape[0])).to(x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2:]
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(h * cut), int(w * cut)
        cy, cx = int(rng.integers(h)), int(rng.integers(w))
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, h)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, w)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (h * w)
    else:
        raise ValueError(f"mode không hợp lệ: {mode}")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)."""
    y_a, y_b, lam = targets
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
