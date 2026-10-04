"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (giữ phần đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
    weight_tag(model) -> str (tag trọng số timm thực sự được tải)
"""
from __future__ import annotations

# Tag trọng số timm có thể đổi theo phiên bản; tag thực sự được tải được ghi vào config.json của mỗi lần chạy.
SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    """Tạo model phân loại 9 lớp. init: "scratch" | "frozen" | "finetune" (trục A)."""
    import timm

    if init not in ("scratch", "frozen", "finetune"):
        raise ValueError(f"init không hợp lệ: {init}")
    name = SUGGESTED_BACKBONES.get(name, name)
    model = timm.create_model(name, pretrained=(pretrained and init != "scratch"),
                              num_classes=num_classes, drop_rate=drop_rate)
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    """Tag trọng số (ví dụ 'resnet50.a1_in1k'); 'random-init' nếu train từ đầu."""
    if getattr(model, "init_mode", "finetune") == "scratch":
        return "random-init"
    cfg = getattr(model, "pretrained_cfg", {}) or {}
    arch = cfg.get("architecture", "?")
    tag = cfg.get("tag", "")
    return f"{arch}.{tag}" if tag else arch


def _head_param_ids(model) -> set[int]:
    return {id(p) for p in model.get_classifier().parameters()}


def freeze_backbone(model) -> None:
    """Đóng băng mọi tham số trừ head. BatchNorm của backbone được giữ ở eval bởi set_train_mode()."""
    head = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head
    model.backbone_frozen = True


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone đóng băng thì mọi module ngoài head trở về eval.

    Lý do: BatchNorm ở train mode vẫn cập nhật running_mean/var (không cần gradient) và chuẩn hoá theo thống kê
    batch, làm đặc trưng của backbone "đóng băng" thay đổi theo từng batch.
    """
    model.train()
    if getattr(model, "backbone_frozen", False):
        head = model.get_classifier()
        head_modules = set(head.modules())
        for m in model.modules():
            if m not in head_modules and m is not model:
                m.eval()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm tham số (slide trang 52):
      - backbone ndim > 1      : lr_backbone, weight_decay
      - backbone norm/bias     : lr_backbone, weight_decay = 0
      - head mới               : lr_head, weight_decay (bias của head cũng không decay)
    """
    head = _head_param_ids(model)
    groups = {"backbone_decay": [], "backbone_no_decay": [], "head_decay": [], "head_no_decay": []}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        part = "head" if id(p) in head else "backbone"
        # norm, bias, và các token/embedding vị trí 1-D của transformer: không weight decay
        no_decay = p.ndim <= 1 or name.endswith(".bias") or "pos_embed" in name or "cls_token" in name \
            or "relative_position_bias_table" in name
        groups[f"{part}_{'no_decay' if no_decay else 'decay'}"].append(p)
    out = []
    for key, params in groups.items():
        if not params:
            continue
        out.append({"params": params, "name": key,
                    "lr": lr_head if key.startswith("head") else lr_backbone,
                    "weight_decay": 0.0 if key.endswith("no_decay") else weight_decay})
    return out


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số bị đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho một ảnh 3 x img_size x img_size, đếm bằng fvcore (fvcore đếm 1 MAC = 1 "flop").

    Chạy trên bản sao CPU để không ảnh hưởng model đang train. Số có thể lệch vài % so với công cụ khác
    (fvcore bỏ qua các phép elementwise như activation, softmax).
    """
    import copy
    import logging

    import torch
    from fvcore.nn import FlopCountAnalysis

    logging.getLogger("fvcore").setLevel(logging.ERROR)
    m = copy.deepcopy(model).to("cpu").float().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    with torch.no_grad():
        fca = FlopCountAnalysis(m, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        total = fca.total()
    return total / 1e9
