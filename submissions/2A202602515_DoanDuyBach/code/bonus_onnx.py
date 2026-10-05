"""Điểm thưởng (RUBRIC mục 2): xuất chung kết F01 seed 0 (DeiT-S, ảnh đầy đủ 288) sang ONNX và so sánh độ trễ với PyTorch
trên cùng máy (Apple M5): batch 1, FP32, warmup 10, 100 lần đo, không tính tiền xử lý. PyTorch MPS có
torch.mps.synchronize trước/sau mỗi lần đo; PyTorch CPU và ONNX Runtime chạy đồng bộ. Kiểm tra tương đương trên VAL fold 0
(không đụng test): sai khác logit lớn nhất, tỉ lệ trùng nhãn đoán và macro-F1 của từng backend.

Mô hình ONNX có kích thước đầu vào tĩnh (1, 3, 288, 288), đúng kịch bản robot xử lý từng khung hình: CoreML EP không chạy
được batch động ("unbounded dimension"). Backend: ONNX Runtime CPU EP và CoreML EP (định dạng MLProgram, MLComputeUnits=ALL).

Chạy khi máy rảnh (không train song song) để số đo CPU không bị nhiễu:
    python code/bonus_onnx.py   (từ thư mục bài nộp) -> logs/bonus_onnx.json; file .onnx ở runs/ (không commit)
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train as TR  # noqa: E402
import dataset as D  # noqa: E402
from benchmark import bench, device_name  # noqa: E402
from eval import compute_metrics  # noqa: E402
from step3_inference import load_model, loader_for  # noqa: E402

SIZE = 288
ONNX_PATH = Path("runs/F01/seed0/f01_seed0_288_b1.onnx")
PROVIDERS = {"onnxruntime_cpu": ["CPUExecutionProvider"],
             "onnxruntime_coreml": [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                                    "CPUExecutionProvider"]}


def export(model):
    import onnx
    import torch

    torch.onnx.export(model, (torch.randn(1, 3, SIZE, SIZE),), str(ONNX_PATH), input_names=["input"],
                      output_names=["logits"], opset_version=18, dynamo=True)
    onnx.checker.check_model(str(ONNX_PATH))
    return sum(p.stat().st_size for p in ONNX_PATH.parent.glob(ONNX_PATH.name + "*")) / 1e6


def main():
    import onnx
    import onnxruntime as ort
    import torch

    out = {"model": "F01 seed 0, deit_small_patch16_224 (dynamic_img_size), ảnh đầy đủ 288, FP32, đầu vào tĩnh batch 1",
           "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
           "device": device_name("mps"), "providers": {k: str(v) for k, v in PROVIDERS.items()},
           "latency_b1": {}, "parity_val": {}}
    cpu_model = load_model("F01", "deit_small_patch16_224", 0, torch.device("cpu"))
    out["onnx_size_mb"] = export(cpu_model)
    mps_model = load_model("F01", "deit_small_patch16_224", 0, torch.device("mps"))
    sessions = {k: ort.InferenceSession(str(ONNX_PATH), providers=v) for k, v in PROVIDERS.items()}

    # tương đương trên VAL (ảnh đầy đủ 288, đúng tiền xử lý của chung kết); ONNX chạy từng ảnh vì đầu vào tĩnh batch 1
    _, va, _ = D.load_split("../../data/labels", 0)
    zs = {k: [] for k in ("pytorch_mps", "pytorch_cpu", *sessions)}
    ys = []
    for x, y, _ in loader_for(va, "../../data/images", SIZE, False):
        with torch.inference_mode():
            zs["pytorch_mps"].append(mps_model(x.to("mps")).float().cpu().numpy())
            zs["pytorch_cpu"].append(cpu_model(x).float().numpy())
        xn = x.numpy()
        for k, s in sessions.items():
            zs[k].append(np.concatenate([s.run(None, {"input": xn[i:i + 1]})[0] for i in range(len(xn))]))
        ys.append(y.numpy())
    y = np.concatenate(ys)
    ref = np.concatenate(zs["pytorch_cpu"])
    for k, v in zs.items():
        z = np.concatenate(v)
        p = TR.softmax_np(z)
        r = compute_metrics(y, p.argmax(1), p)
        out["parity_val"][k] = {"macro_f1": r["macro_f1"], "top1": r["top1"],
                                "max_abs_logit_diff_vs_pytorch_cpu": float(np.abs(z - ref).max()),
                                "argmax_agreement_vs_pytorch_cpu": float((z.argmax(1) == ref.argmax(1)).mean())}
        print(k, out["parity_val"][k], flush=True)

    # độ trễ batch 1
    x = torch.randn(1, 3, SIZE, SIZE)
    xm, xn = x.to("mps"), x.numpy()

    def run_cpu():
        with torch.inference_mode():
            cpu_model(x)

    def run_mps():
        with torch.inference_mode():
            mps_model(xm)

    out["latency_b1"]["pytorch_mps"] = bench(run_mps, 10, 100, torch.mps.synchronize)
    out["latency_b1"]["pytorch_cpu"] = bench(run_cpu, 10, 100)
    for k, s in sessions.items():
        out["latency_b1"][k] = bench(lambda: s.run(None, {"input": xn}), 10, 100)
    out["torch_cpu_threads"] = torch.get_num_threads()
    for k, v in out["latency_b1"].items():
        print(k, {q: round(v[q], 2) for q in ("p50", "p95", "p99")}, flush=True)
    Path("logs/bonus_onnx.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
