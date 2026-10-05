"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Không cần dữ liệu, chạy trên CPU:

    cd code && python -m unittest test_code -v   (tên thư mục "code" trùng module chuẩn nên chạy từ trong code/)
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import losses as L  # noqa: E402
import inference as I  # noqa: E402
import model as M  # noqa: E402
import train as TR  # noqa: E402
import benchmark as B  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.logits = torch.randn(64, 9, generator=g) * 3
        self.y = torch.randint(0, 9, (64,), generator=g)

    def test_focal_gamma0_equals_ce(self):
        fl = L.FocalLoss(gamma=0.0)(self.logits, self.y)
        ce = F.cross_entropy(self.logits, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy_examples(self):
        self.assertLess(L.FocalLoss(2.0)(self.logits, self.y).item(), F.cross_entropy(self.logits, self.y).item())

    def test_focal_alpha_matches_weighted_sum(self):
        alpha = torch.rand(9)
        fl = L.FocalLoss(gamma=0.0, alpha=alpha)(self.logits, self.y)
        ref = (F.cross_entropy(self.logits, self.y, reduction="none") * alpha[self.y]).mean()
        self.assertLess(abs(fl.item() - ref.item()), 1e-6)

    def test_label_smoothing_eps0_equals_ce_and_matches_torch(self):
        self.assertLess(abs(L.LabelSmoothingCE(0.0)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y).item()), 1e-6)
        self.assertLess(abs(L.LabelSmoothingCE(0.1)(self.logits, self.y).item()
                            - F.cross_entropy(self.logits, self.y, label_smoothing=0.1).item()), 1e-5)

    def test_class_weights(self):
        counts = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
        w = L.class_weights(counts)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertLess(w[8].item(), w[0].item())  # Negatives nhiều ảnh -> trọng số nhỏ
        np.testing.assert_allclose((w * torch.tensor(counts, dtype=torch.float32)).numpy(),
                                   (w[0] * counts[0]).item(), rtol=1e-5)  # w ∝ 1/n
        wb = L.class_weights(counts, beta=0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)

    def test_build_criterion(self):
        for kind, kw in [("ce", {}), ("ls", {"smoothing": 0.1}), ("focal", {"gamma": 2}),
                         ("ce_weighted", {"weight": torch.ones(9)})]:
            loss = L.build_criterion(kind, **kw)(self.logits, self.y)
            self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(L.build_criterion("ce_weighted", weight=torch.ones(9))(self.logits, self.y).item(),
                               F.cross_entropy(self.logits, self.y).item(), places=5)


class TestMix(unittest.TestCase):
    def test_cutmix_lambda_equals_true_area(self):
        # mỗi ảnh i có giá trị pixel = i; sau CutMix, tỉ lệ pixel còn giá trị gốc phải đúng bằng lam
        rng = np.random.default_rng(0)
        n_checked = 0
        for _ in range(100):
            x = torch.arange(8).float().view(8, 1, 1, 1).expand(8, 3, 30, 40).clone()
            y = torch.arange(8)
            xm, (ya, yb, lam) = L.mix_batch(x, y, 1.0, "cutmix", rng)
            torch.testing.assert_close(ya, y)
            for i in range(8):
                if yb[i] != ya[i]:
                    kept = (xm[i] == i).float().mean().item()
                    self.assertAlmostEqual(kept, lam, places=6)
                    n_checked += 1
        self.assertGreater(n_checked, 100)

    def test_mixup_mixes_inputs_and_labels(self):
        rng = np.random.default_rng(1)
        x = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
        xm, (ya, yb, lam) = L.mix_batch(x, torch.tensor([0, 1]), 0.4, "mixup", rng)
        self.assertTrue(0.0 <= lam <= 1.0)
        for i in range(2):
            expected = lam * x[i] + (1 - lam) * x[[0, 1].index(int(yb[i]))]
            torch.testing.assert_close(xm[i], expected)

    def test_mixed_loss_is_convex_combination(self):
        logits = torch.randn(4, 9)
        ya, yb = torch.tensor([0, 1, 2, 3]), torch.tensor([3, 2, 1, 0])
        ce = nn.CrossEntropyLoss()
        v = L.mixed_loss(ce, logits, (ya, yb, 0.3))
        self.assertAlmostEqual(v.item(), 0.3 * ce(logits, ya).item() + 0.7 * ce(logits, yb).item(), places=5)


class TestModel(unittest.TestCase):
    def test_param_groups_no_decay_for_norm_and_bias(self):
        m = M.build_model("resnet18", pretrained=False, init="finetune")
        groups = M.param_groups(m, 1e-4, 1e-3, 0.05)
        names = {id(p): n for n, p in m.named_parameters()}
        head = {id(p) for p in m.get_classifier().parameters()}
        n_total = 0
        for g in groups:
            n_total += len(g["params"])
            for p in g["params"]:
                if p.ndim <= 1:
                    self.assertEqual(g["weight_decay"], 0.0, names[id(p)])
                self.assertEqual(g["lr"], 1e-3 if id(p) in head else 1e-4, names[id(p)])
        self.assertEqual(n_total, len(list(m.parameters())))

    def test_frozen_keeps_bn_in_eval_and_only_head_trains(self):
        m = M.build_model("resnet18", pretrained=False, init="finetune")
        M.freeze_backbone(m)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        self.assertEqual(sorted(trainable), ["fc.bias", "fc.weight"])
        M.set_train_mode(m)
        self.assertFalse(m.bn1.training)
        self.assertTrue(m.fc.training)
        rm = m.bn1.running_mean.clone()
        m(torch.randn(4, 3, 64, 64))
        torch.testing.assert_close(m.bn1.running_mean, rm)

    def test_gmacs_resnet50(self):
        g = M.count_gmacs(M.build_model("resnet50", pretrained=False), 224)
        self.assertAlmostEqual(g, 4.1, delta=0.15)


class TestInference(unittest.TestCase):
    def _bn_model(self):
        m = M.build_model("resnet18", pretrained=False)
        # BN ngẫu nhiên để việc gộp không tầm thường
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_mean.uniform_(-0.5, 0.5)
                mod.running_var.uniform_(0.5, 2.0)
                mod.weight.data.uniform_(0.5, 1.5)
                mod.bias.data.uniform_(-0.2, 0.2)
        return m.eval()

    def test_fuse_conv_bn_resnet(self):
        m = self._bn_model()
        fused = I.fuse_conv_bn(m)
        self.assertGreater(fused.n_fused_bn, 15)
        self.assertEqual(sum(isinstance(x, nn.BatchNorm2d) for x in fused.modules()), 0)
        x = torch.randn(2, 3, 96, 96)
        self.assertLess(I.max_abs_diff(m, fused, x), 1e-4)

    def test_fuse_conv_bn_efficientnet(self):
        m = M.build_model("efficientnet_b0", pretrained=False).eval()
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
        fused = I.fuse_conv_bn(m)
        self.assertEqual(sum(isinstance(x, nn.BatchNorm2d) for x in fused.modules()), 0)
        self.assertLess(I.max_abs_diff(m, fused, torch.randn(2, 3, 96, 96)), 1e-4)

    def test_recalibrate_bn_matches_data_statistics(self):
        m = self._bn_model()
        g = torch.Generator().manual_seed(0)
        batches = [(torch.randn(8, 3, 64, 64, generator=g) * 2 + 1, torch.zeros(8)) for _ in range(4)]
        r = I.recalibrate_bn(m, batches, torch.device("cpu"))
        with torch.no_grad():
            feats = torch.cat([m.conv1(x) for x, _ in batches])
        torch.testing.assert_close(r.bn1.running_mean, feats.mean((0, 2, 3)), atol=1e-4, rtol=1e-4)
        self.assertFalse(r.training)
        self.assertFalse(torch.allclose(m.bn1.running_mean, r.bn1.running_mean))  # bản gốc không đổi

    def test_aggregate_and_ensemble(self):
        rng = np.random.default_rng(0)
        views = [rng.normal(size=(10, 9)) for _ in range(3)]
        for space in ("prob", "logit"):
            p = I.aggregate_views(views, space)
            np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-9)
        np.testing.assert_allclose(I.aggregate_views([views[0]], "prob"), I._softmax(views[0]))
        e = I.ensemble_probs([I._softmax(v) for v in views])
        np.testing.assert_allclose(e.sum(1), 1.0, atol=1e-9)

    def test_temperature_recovers_known_T(self):
        rng = np.random.default_rng(0)
        n, true_T = 5000, 2.5
        z = rng.normal(size=(n, 9)) * 4
        p = I._softmax(z / true_T)
        y = np.array([rng.choice(9, p=pi) for pi in p])
        T = I.fit_temperature(z, y)
        self.assertAlmostEqual(T, true_T, delta=0.25)
        np.testing.assert_array_equal(I.apply_temperature(z, T).argmax(1), z.argmax(1))  # accuracy không đổi

    def test_views(self):
        x = torch.randn(2, 3, 256, 256)
        self.assertEqual(len(I.views_multicrop(x, 224)), 5)
        self.assertEqual(len(I.views_multicrop(x, 224, flip=True)), 10)
        self.assertEqual([v.shape[-1] for v in I.views_multiscale(x, [224, 256, 288])], [224, 256, 288])
        torch.testing.assert_close(I.view_hflip(I.view_hflip(x)), x)


class TestTrainHelpers(unittest.TestCase):
    def test_lr_schedule_shape(self):
        total, warm = 100, 10
        f = [TR.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[9], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(f[:10], f[1:10])))   # tăng khi warmup
        self.assertTrue(all(a >= b for a, b in zip(f[10:], f[11:])))    # giảm khi cosine
        self.assertLess(f[-1], 0.01)

    def test_parse_overrides(self):
        d = TR.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "amp=false", "lr_head=3e-3",
                                "sampler=balanced"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "amp": False, "lr_head": 3e-3,
                             "sampler": "balanced"})
        with self.assertRaises(KeyError):
            TR.parse_overrides(["nope=1"])

    def test_ema_tracks_weights(self):
        m = nn.Linear(3, 2)
        ema = TR.EMA(m, 0.9)
        with torch.no_grad():
            m.weight.add_(1.0)
        for _ in range(200):
            ema.update(m)
        torch.testing.assert_close(ema.module.weight, m.weight, atol=1e-4, rtol=0)

    def test_bench(self):
        r = B.bench(lambda: sum(range(1000)), warmup=3, iters=50)
        self.assertEqual(r["n"], 50)
        self.assertLessEqual(r["p50"], r["p95"])
        self.assertLessEqual(r["p95"], r["p99"])


class TestBonus(unittest.TestCase):
    """Các hàm của phần điểm thưởng (bonus_shift_tta.py, bonus_attention.py)."""

    def setUp(self):
        from PIL import Image

        rng = np.random.default_rng(0)
        self.img = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))

    def test_corrupt_clean_dark_noise(self):
        import bonus_shift_tta as S

        a = np.asarray(self.img, dtype=np.float32)
        self.assertIs(S.corrupt(self.img, None, None, 0), self.img)
        dark = np.asarray(S.corrupt(self.img, "dark", 0.5, 0), dtype=np.float32)
        self.assertLess(np.abs(dark - a * 0.5).max(), 1.0)  # chỉ sai số làm tròn 8 bit
        n1 = np.asarray(S.corrupt(self.img, "noise", 0.1, 7))
        n2 = np.asarray(S.corrupt(self.img, "noise", 0.1, 7))
        self.assertTrue((n1 == n2).all())  # nhiễu cố định theo seed của ảnh
        self.assertFalse((n1 == np.asarray(S.corrupt(self.img, "noise", 0.1, 8))).all())

    def test_rollout_uniform_attention(self):
        import bonus_attention as A

        n = 1 + 16  # CLS + 4 × 4 patch
        attn = [torch.full((2, 3, n, n), 1.0 / n) for _ in range(4)]
        r = A.rollout(attn, 1)
        self.assertEqual(tuple(r.shape), (2, 16))
        self.assertTrue(torch.allclose(r, torch.full_like(r, 1 / 16), atol=1e-6))

    def test_otsu_splits_bimodal(self):
        import bonus_attention as A

        v = np.concatenate([np.full(500, -0.3), np.full(500, 0.4)]) + np.random.default_rng(0).normal(0, .02, 1000)
        t = A.otsu(v)
        self.assertTrue(-0.2 < t < 0.3)


if __name__ == "__main__":
    unittest.main()
