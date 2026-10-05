"""Điểm thưởng (RUBRIC mục 2): bản đồ attention và Grad-CAM của chung kết F01 seed 0 (DeiT-S, ảnh đầy đủ 288) để giải thích
lỗi Chinee apple ↔ Snake weed.

- Attention rollout (Abnar & Zuidema, 2020): trung bình các head, cộng ma trận đơn vị (kết nối tắt), chuẩn hoá theo hàng,
  nhân dồn qua 12 block, lấy hàng của token CLS trên 18 × 18 patch. Không phụ thuộc lớp.
- Grad-CAM cho ViT: kích hoạt và gradient tại LayerNorm đầu của block cuối (norm1, đầu vào của lớp attention cuối cùng
  trộn patch vào CLS), trọng số kênh = trung bình gradient theo patch, bản đồ = ReLU(tổng có trọng số). Gắn với lớp được
  đoán, nên chỉ ra vùng làm mô hình chọn nhãn (sai).
- Định tính: các ảnh TEST bị nhầm giữa hai lớp, lấy đúng từ file dự đoán của Bước 4 (predictions/F01_seed0_test.csv),
  kèm 2 ảnh đoán đúng mỗi lớp. Không tính lại chỉ số test nào; script kiểm tra nhãn đoán khớp với file dự đoán.
- Định lượng (trên VAL, ảnh Chinee apple và Snake weed): tỉ lệ khối lượng bản đồ rơi vào điểm ảnh thực vật xanh chia cho tỉ
  lệ diện tích thực vật xanh (> 1: tập trung vào cây), và entropy chuẩn hoá của bản đồ (1 = trải đều). Thực vật xanh =
  ExG > ngưỡng Otsu của từng ảnh, ExG = 2g − r − b với r, g, b là toạ độ màu chuẩn hoá.

    python code/bonus_attention.py   (từ thư mục bài nộp) -> figures/gradcam_chinee_snake.png,
                                      figures/attention_rollout_chinee_snake.png, logs/bonus_attention.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import train as TR  # noqa: E402
import dataset as D  # noqa: E402
from step3_inference import load_model  # noqa: E402

IMAGES, SIZE = Path("../../data/images"), 288
NAMES = {0: "Chinee apple", 7: "Snake weed"}


def attach(model):
    """Hook: ma trận attention sau softmax của mọi block (tắt fused attention) và kích hoạt norm1 của block cuối."""
    store = {"attn": [], "act": None}
    for blk in model.blocks:
        blk.attn.fused_attn = False
        blk.attn.attn_drop.register_forward_hook(lambda mod, inp, out: store["attn"].append(out.detach()))

    def keep(mod, inp, out):
        out.retain_grad()
        store["act"] = out

    model.blocks[-1].norm1.register_forward_hook(keep)
    return store


def rollout(attns, n_prefix):
    import torch

    joint = None
    for a in attns:
        a = a.float().mean(1)
        a = a + torch.eye(a.shape[-1], device=a.device)
        a = a / a.sum(-1, keepdim=True)
        joint = a if joint is None else a @ joint
    cls = joint[:, 0, n_prefix:]
    return cls / cls.sum(-1, keepdim=True)


def run(model, store, names, device, bs=8):
    """-> (ảnh RGB 288 [0,1], logits, rollout 18 × 18, Grad-CAM 18 × 18 của lớp được đoán) cho danh sách tên file."""
    import torch
    from PIL import Image
    from torchvision import transforms as T

    tf = T.Compose([T.Resize((SIZE, SIZE)), T.ToTensor()])
    norm = T.Normalize(D.IMAGENET_MEAN, D.IMAGENET_STD)
    imgs, logits, rolls, cams = [], [], [], []
    n_prefix = model.num_prefix_tokens
    for i in range(0, len(names), bs):
        x = torch.stack([tf(Image.open(IMAGES / n).convert("RGB")) for n in names[i:i + bs]])
        store["attn"].clear()
        model.zero_grad(set_to_none=True)
        z = model(norm(x).to(device).requires_grad_(True))  # gradient đi qua kích hoạt; trọng số vẫn đóng băng
        z.gather(1, z.argmax(1, keepdim=True)).sum().backward()
        act, grad = store["act"][:, n_prefix:], store["act"].grad[:, n_prefix:]
        cam = torch.relu((grad.mean(1, keepdim=True) * act).sum(-1)).detach()
        cam = cam / cam.sum(-1, keepdim=True).clamp_min(1e-12)
        side = int(round(cam.shape[-1] ** 0.5))
        rolls.append(rollout(store["attn"], n_prefix).reshape(-1, side, side).cpu().numpy())
        cams.append(cam.reshape(-1, side, side).cpu().numpy())
        imgs.append(x.permute(0, 2, 3, 1).numpy())
        logits.append(z.detach().float().cpu().numpy())
    return np.concatenate(imgs), np.concatenate(logits), np.concatenate(rolls), np.concatenate(cams)


def upsample(m):
    import torch
    import torch.nn.functional as F

    return F.interpolate(torch.as_tensor(m)[None, None], size=(SIZE, SIZE), mode="bilinear",
                         align_corners=False)[0, 0].numpy()


def otsu(v):
    """Ngưỡng Otsu = cận trên của bin cuối thuộc lớp dưới (cực đại phương sai giữa hai lớp);
    nếu nhiều bin cùng đạt cực đại (khoảng trống giữa hai cụm) thì lấy điểm giữa của chúng."""
    hist, edges = np.histogram(v, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    c = np.cumsum(hist * centers)
    m0 = c / np.maximum(w0, 1)
    m1 = (c[-1] - c) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    best = np.flatnonzero(between >= between.max() * (1 - 1e-9))
    return float(edges[1:][best].mean())


def veg_mask(img):
    s = img.sum(-1) + 1e-6
    exg = 2 * img[..., 1] / s - img[..., 0] / s - img[..., 2] / s
    return exg > otsu(exg)


def map_stats(amap, mask):
    a = upsample(amap).clip(min=0)
    p = amap.ravel() / max(amap.sum(), 1e-12)
    ent = float(-(p * np.log(p + 1e-12)).sum() / np.log(p.size))
    return float((a * mask).sum() / max(a.sum(), 1e-12) / max(mask.mean(), 1e-6)), ent


def figure(pick, imgs, maps, title, path):
    cols = 5
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.4 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for k, (ax, (_, row)) in enumerate(zip(axes.ravel(), pick.iterrows())):
        ax.imshow(imgs[k])
        ax.imshow(upsample(maps[k]), cmap="jet", alpha=0.45)
        wrong = row.y_true != row.y_pred
        ax.set_title(f"{'SAI' if wrong else 'đúng'}: thật {NAMES[row.y_true]}\nđoán {NAMES[row.y_pred]} ({row.conf:.2f})",
                     fontsize=8, color="red" if wrong else "green")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    device = TR.get_device()
    model = load_model("F01", "deit_small_patch16_224", 0, device)
    for p in model.parameters():
        p.requires_grad_(False)  # chỉ cần gradient theo kích hoạt, không theo trọng số
    store = attach(model)

    # 1) định tính: ảnh test bị nhầm (từ file dự đoán Bước 4) + 2 ảnh đúng mỗi lớp
    d = pd.read_csv("predictions/F01_seed0_test.csv")
    d["conf"] = d[[f"p{k}" for k in range(9)]].max(1)
    err = d[((d.y_true == 0) & (d.y_pred == 7)) | ((d.y_true == 7) & (d.y_pred == 0))].sort_values(
        ["y_true", "conf"], ascending=[True, False])
    ok = pd.concat([d[(d.y_true == c) & (d.y_pred == c)].nlargest(2, "conf") for c in (0, 7)])
    pick = pd.concat([err, ok])
    imgs, z, rolls, cams = run(model, store, pick.Filename.tolist(), device)
    n_mismatch = int((z.argmax(1) != pick.y_pred.to_numpy()).sum())  # kỳ vọng 0 (cùng checkpoint, cùng tiền xử lý)
    print("số ảnh có nhãn đoán khác file dự đoán Bước 4:", n_mismatch)
    head = "F01 seed 0 (DeiT-S, ảnh đầy đủ 288), ảnh test lấy từ dự đoán Bước 4: "
    figure(pick, imgs, cams, head + "Grad-CAM của lớp được đoán (norm1 block cuối)", "figures/gradcam_chinee_snake.png")
    figure(pick, imgs, rolls, head + "attention rollout của token CLS", "figures/attention_rollout_chinee_snake.png")
    qual = [{"file": r.Filename, "y_true": int(r.y_true), "y_pred": int(r.y_pred), "conf": float(r.conf),
             "veg_area": float(veg_mask(imgs[k]).mean()),
             "gradcam_veg_ratio": map_stats(cams[k], veg_mask(imgs[k]))[0]} for k, (_, r) in enumerate(pick.iterrows())]

    # 2) định lượng trên VAL: bản đồ có tập trung vào thực vật xanh không, ảnh đúng so với ảnh sai
    _, va, _ = D.load_split("../../data/labels", 0)
    v = va[va.Label.isin([0, 7])].reset_index(drop=True)
    imgs, z, rolls, cams = run(model, store, v.Filename.tolist(), device)
    rows = []
    for k in range(len(v)):
        mask = veg_mask(imgs[k])
        rr, re = map_stats(rolls[k], mask)
        cr, ce = map_stats(cams[k], mask)
        rows.append({"correct": bool(z[k].argmax() == v.Label[k]), "veg_area": float(mask.mean()),
                     "rollout_veg_ratio": rr, "rollout_entropy": re, "gradcam_veg_ratio": cr, "gradcam_entropy": ce})
    df = pd.DataFrame(rows)
    summ = {("dung" if c else "sai"): {"n": int(len(g)), **{k: {"mean": float(g[k].mean()), "std": float(g[k].std())}
                                                           for k in df.columns if k != "correct"}}
            for c, g in df.groupby("correct")}
    out = {"qualitative_test": qual, "n_test_errors": int(len(err)), "n_pred_mismatch_vs_step4": n_mismatch,
           "val_quantitative": summ, "val_classes": "Chinee apple + Snake weed (val fold 0)",
           "note": "veg_ratio > 1: bản đồ tập trung vào thực vật xanh hơn mức trải đều; entropy chuẩn hoá về [0, 1]"}
    Path("logs/bonus_attention.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
