"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Liên hệ slide Day 2: TTA (trang 62-66, 75), ensemble/EMA/soup (trang 67), độ phân giải kiểm tra
(trang 68), temperature scaling (trang 69), gộp BatchNorm (trang 71).

Mọi hàm chạy ở chế độ eval, không gradient. Chọn phương pháp CHỈ dựa trên val;
nhiệt độ T khớp trên VAL rồi áp dụng sang test (README.md, S2 và S4).

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, list[logits[N, 9]])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def predict_logits(model, loader, device, view=None, amp_dtype=None):
    """Chạy model trên loader, gom logit theo đúng thứ tự file. `view`: hàm biến đổi batch (hoặc None)."""
    names, ys, logits_per_view = predict_views(model, loader, device,
                                               (lambda x: [view(x)]) if view is not None else None, amp_dtype)
    return names, ys, logits_per_view[0]


def predict_views(model, loader, device, views_fn=None, amp_dtype=None):
    """Như predict_logits, nhưng `views_fn(x) -> list[batch]` trả về K view; kết quả là list K mảng logit."""
    model.eval()
    names, ys, outs = [], [], None
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            views = views_fn(x) if views_fn is not None else [x]
            if outs is None:
                outs = [[] for _ in views]
            for k, v in enumerate(views):
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    outs[k].append(model(v).float().cpu())
            ys.append(y)
            names.extend(f)
    return names, torch.cat(ys).numpy(), [torch.cat(o).numpy() for o in outs]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W) theo chiều rộng (slide trang 75)."""
    return torch.flip(x, dims=[3])


def views_hflip(x):
    """TTA K = 2: ảnh gốc + lật ngang."""
    return [x, view_hflip(x)]


def views_multicrop(x, crop: int, flip: bool = False):
    """5 crop (4 góc + giữa) kích thước `crop` từ batch x; flip=True thêm bản lật của mỗi crop (K = 10)."""
    h, w = x.shape[-2:]
    if crop > h or crop > w:
        raise ValueError(f"crop {crop} lớn hơn ảnh {h}x{w}")
    top, left = (h - crop) // 2, (w - crop) // 2
    crops = [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
             x[..., h - crop:, w - crop:], x[..., top:top + crop, left:left + crop]]
    if flip:
        crops = crops + [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes):
    """Resize batch về từng kích thước trong `sizes` (bilinear, antialias). CNN có global pooling chạy được
    với mọi kích thước; ViT/Swin cần nội suy pos-embed / đổi cửa sổ nên KHÔNG dùng hàm này cho chúng."""
    return [x if s == x.shape[-1] else F.interpolate(x, size=(s, s), mode="bilinear", align_corners=False,
                                                     antialias=True) for s in sizes]


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax. Trả về probs (N, 9)."""
    arr = np.stack([np.asarray(v, dtype=np.float64) for v in logits_per_view])
    if space == "prob":
        p = np.stack([_softmax(v) for v in arr]).mean(0)
    elif space == "logit":
        p = _softmax(arr.mean(0))
    else:
        raise ValueError(f"space không hợp lệ: {space}")
    return p / p.sum(1, keepdims=True)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    if arr.ndim != 3:
        raise ValueError("mỗi phần tử phải là mảng (N, K)")
    p = arr.mean(0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels) -> float:
    """T > 0 cực tiểu NLL trên VAL của softmax(logit / T). Tìm lưới log T thô rồi LBFGS trên log T."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = torch.linspace(np.log(0.05), np.log(20.0), 200, dtype=torch.float64)
    nll = torch.stack([F.cross_entropy(z / t.exp(), y) for t in grid])
    log_t = grid[nll.argmin()].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.detach().exp())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return _softmax(np.asarray(logits, dtype=np.float64) / T)


def _fuse(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride, conv.padding,
                      conv.dilation, conv.groups, bias=True, padding_mode=conv.padding_mode)
    fused = fused.to(conv.weight.device, conv.weight.dtype)
    std = torch.sqrt(bn.running_var + bn.eps)
    gamma = bn.weight if bn.affine else torch.ones_like(std)
    beta = bn.bias if bn.affine else torch.zeros_like(std)
    scale = gamma / std
    b = conv.bias if conv.bias is not None else torch.zeros_like(bn.running_mean)
    with torch.no_grad():
        fused.weight.copy_(conv.weight * scale.reshape(-1, 1, 1, 1))
        fused.bias.copy_(beta + (b - bn.running_mean) * scale)
    return fused


def fuse_conv_bn(model):
    """Gộp BatchNorm2d vào Conv2d liền trước (trong cùng module cha, theo thứ tự đăng ký), trả về bản sao.

        w' = gamma * w / sqrt(var + eps)        b' = beta + gamma * (b - mean) / sqrt(var + eps)

    Duyệt từng module cha: Conv2d đứng ngay trước BatchNorm2d (theo thứ tự con) được gộp, BN thay bằng Identity.
    Đúng với ResNet/ResNeXt/EfficientNet/MobileNet của timm (conv1->bn1, conv->bn, ...). Với timm, BN của
    EfficientNet/MobileNet là BatchNormAct2d (BN + activation): ta gộp phần BN và giữ activation.
    Kiến trúc không có BN (ViT, Swin, ConvNeXt dùng LayerNorm): không áp dụng, hàm trả về bản sao không đổi.
    """
    model = copy.deepcopy(model).eval()
    n_fused = 0

    def bn_core(m):
        return isinstance(m, nn.BatchNorm2d)

    for parent in list(model.modules()):
        children = list(parent.named_children())
        for (n1, m1), (n2, m2) in zip(children, children[1:]):
            if isinstance(m1, nn.Conv2d) and bn_core(m2) and m1.out_channels == m2.num_features:
                setattr(parent, n1, _fuse(m1, m2))
                act = None
                # timm BatchNormAct2d: giữ drop + act sau khi bỏ phần chuẩn hoá
                if hasattr(m2, "act") and hasattr(m2, "drop"):
                    act = nn.Sequential(m2.drop, m2.act)
                setattr(parent, n2, act if act is not None else nn.Identity())
                n_fused += 1
    model.n_fused_bn = n_fused
    return model


def max_abs_diff(model_a, model_b, x) -> float:
    """Sai số lớn nhất giữa đầu ra của hai model trên cùng batch (dùng để kiểm tra gộp BN)."""
    with torch.inference_mode():
        return float((model_a.eval()(x) - model_b.eval()(x)).abs().max())


def recalibrate_bn(model, loader, device, max_batches: int | None = None):
    """Precise-BN: tính lại running_mean/var của mọi BatchNorm bằng ảnh TRAIN đi qua transform của val/test
    (không augmentation), trả về bản sao. Không dùng nhãn, không dùng val/test nên hợp lệ với S4.

    Lý do: lúc train, RandomResizedCrop phóng to vật thể nên thống kê BN tích luỹ khác với ảnh CenterCrop lúc
    đánh giá (lệch train/test kiểu FixRes). momentum=None -> trung bình cộng dồn trên mọi batch.
    """
    model = copy.deepcopy(model)
    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return model.eval()
    model.eval()
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
        bn.train()
    with torch.no_grad():
        for i, (x, *_rest) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            model(x.to(device))
    model.eval()
    model.n_recalibrated_bn = len(bns)
    return model
