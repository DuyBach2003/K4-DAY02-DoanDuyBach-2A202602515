"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Quy tắc chia dữ liệu bắt buộc (S1-S6) nằm ở README.md, mục 2.1.

Giao diện:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn tiền xử lý lúc đánh giá: ảnh gốc 256x256 -> Resize(round(img_size / 0.875)) -> CenterCrop(img_size).
Với img_size = 224 thì Resize(256) không đổi ảnh, tức là chỉ CenterCrop(224).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
# Thứ tự lớp theo cột `Label` của labels.csv (0 = Chinee Apple ... 7 = Snake Weed, 8 = Negatives).
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)  # mọi backbone dùng ở đây đều theo mean/std ImageNet (pretrained_cfg)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
EVAL_CROP_PCT = 0.875


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train_subset{fold}.csv, val_subset{fold}.csv, test_subset{fold}.csv (S1). Không sửa, lọc hay chia lại."""
    labels_dir = Path(labels_dir)
    dfs = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        df["Label"] = df["Label"].astype(int)
        dfs.append(df)
    return tuple(dfs)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True) -> dict:
    """Kiểm tra bắt buộc trước khi train (README.md, mục 2.1). Vi phạm thì raise để dừng ngay."""
    images_dir = Path(images_dir)
    splits = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in splits.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(int).tolist()
                 for k, v in splits.items()}

    names = {k: set(v["Filename"]) for k, v in splits.items()}
    for k, v in splits.items():
        assert not v["Filename"].duplicated().any(), f"{k}: có Filename trùng"
    overlap = {"train&val": len(names["train"] & names["val"]),
               "train&test": len(names["train"] & names["test"]),
               "val&test": len(names["val"] & names["test"])}
    assert all(v == 0 for v in overlap.values()), f"giao giữa các tập khác rỗng: {overlap}"
    union = len(names["train"] | names["val"] | names["test"])
    assert union == TOTAL_IMAGES, f"hợp ba tập = {union}, kỳ vọng {TOTAL_IMAGES}"
    for k, target in (("train", 0.6), ("val", 0.2), ("test", 0.2)):
        assert abs(frac[k] - target) < 0.01, f"{k}: tỉ lệ {frac[k]:.4f} lệch > 1 điểm % so với {target}"

    on_disk = {p.name for p in images_dir.glob("*.jpg")}
    missing = sorted((names["train"] | names["val"] | names["test"]) - on_disk)
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}"

    out = {"n": n, "fraction": {k: round(v, 4) for k, v in frac.items()}, "per_class": per_class,
           "overlap": overlap, "union": union, "missing_files": len(missing), "images_on_disk": len(on_disk)}
    if verbose:
        print(f"[check_split] n={n} frac={out['fraction']} union={union} overlap={overlap} "
              f"missing={len(missing)} on_disk={len(on_disk)}")
    return out


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Transform cho train hoặc val/test.

    aug (trục B của GUIDE.md mục 3):
      - "basic"  : RandomResizedCrop + lật ngang
      - "color"  : basic + ColorJitter(0.3, 0.3, 0.3, 0.05)
      - "trivial": basic + TrivialAugmentWide
      - "randaug": basic + RandAugment(num_ops=2, magnitude=9)
      - "flipv"  : basic + lật dọc (ảnh chụp từ trên xuống nên hướng không có nghĩa ngữ nghĩa)
      - "mildcrop": RandomResizedCrop(scale=(0.35, 1)) + lật ngang (crop nhẹ hơn mặc định 0.08-1)
    Mixup/CutMix trộn theo batch nên nằm ở losses.py.
    Val/test: Resize(img_size / 0.875) + CenterCrop(img_size), không augmentation ngẫu nhiên.
    """
    from torchvision import transforms as T

    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return T.Compose([T.Resize(round(img_size / EVAL_CROP_PCT)), T.CenterCrop(img_size), *norm])

    if aug == "mildcrop":
        return T.Compose([T.RandomResizedCrop(img_size, scale=(0.35, 1.0)), T.RandomHorizontalFlip(), *norm])
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    elif aug == "flipv":
        ops.append(T.RandomVerticalFlip())
    else:
        raise ValueError(f"aug không hợp lệ: {aug}")
    return T.Compose([*ops, *norm])


try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # để module import được khi chưa cài torch
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """Đọc ảnh từ `images_dir` theo DataFrame (Filename, Label). __getitem__ -> (tensor, nhãn int, tên file)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.filenames = df["Filename"].astype(str).tolist()
        self.labels = df["Label"].astype(int).tolist()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        from PIL import Image

        name = self.filenames[i]
        with Image.open(self.images_dir / name) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[i], name


def seed_worker(worker_id: int) -> None:
    """Seed cho từng worker của DataLoader (theo seed gốc của torch) để augmentation tái lập được."""
    import torch

    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """DataLoader. train=False: không shuffle, giữ đúng thứ tự df để ghép logit với Filename.

    sampler="balanced": WeightedRandomSampler, trọng số mẫu = 1 / (số ảnh của lớp đó trong df), có hoàn lại,
    số mẫu mỗi epoch = len(df).
    """
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        counts = df["Label"].value_counts()
        w = df["Label"].map(lambda c: 1.0 / counts[c]).to_numpy(dtype=np.float64)
        smp = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(df), replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler không hợp lệ: {sampler}")
    return DataLoader(
        ds, batch_size=batch_size, shuffle=(train and smp is None), sampler=smp,
        drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=g, persistent_workers=num_workers > 0,
    )
