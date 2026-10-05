# Lab Day 2 — DeepWeeds · Đoàn Duy Bách · 2A202602515

Bài nộp Lab Day 2 (backbone, công thức huấn luyện, suy luận trên DeepWeeds, fold 0).
Kết quả và phân tích ở [`report.md`](report.md); bảng số liệu ở [`results.xlsx`](results.xlsx).

## Notebook chạy lại

- Notebook: [`code/lab_day2.ipynb`](code/lab_day2.ipynb)
- Mở trên Colab: <https://colab.research.google.com/github/DuyBach2003/K4-DAY02-DoanDuyBach-2A202602515/blob/main/submissions/2A202602515_DoanDuyBach/code/lab_day2.ipynb>

Notebook clone repo, tải dữ liệu (kiểm tra MD5), rồi gọi đúng các script bên dưới theo thứ tự. Code tự chọn thiết bị
`cuda > mps > cpu`, nên chạy được trên Colab/Kaggle (CUDA) lẫn máy Mac (MPS).

## Môi trường đã dùng để ra số liệu trong bài

| Mục | Giá trị |
|---|---|
| Phần cứng | Apple M5 (10 nhân CPU, GPU 10 nhân, 16 GB RAM), PyTorch backend **MPS** |
| Python | 3.12.14 |
| Thư viện | torch 2.14.1 · torchvision 0.29.1 · timm 1.0.30 · numpy 2.5.3 · pandas 3.0.6 · scikit-learn 1.9.1 · matplotlib 3.11.2 · openpyxl 3.1.5 · fvcore (đếm GMAC) · Pillow 12.3.0 |
| Thư viện thêm cho phần thưởng | onnx 1.23.1 · onnxruntime 1.30.0 (CPU, CoreML EP) · onnxscript 0.7.2; DINOv2 qua timm (`vit_small_patch14_dinov2.lvd142m`) |
| Mixed precision | autocast fp16 + GradScaler (tốc độ train ResNet-50 trên M5, batch 64: fp16 60,9 ảnh/s, bf16 58,4, fp32 44,6; `logs/bench_amp_dtype.json`) |
| Seed | 0 cho Bước 1-3; 0, 1, 2 cho chung kết và mốc |

Cài đặt: `python -m venv .venv && source .venv/bin/activate && pip install torch torchvision timm scikit-learn pandas numpy matplotlib openpyxl fvcore`

## Dữ liệu

`images.zip` (Zenodo 10.5281/zenodo.7939060, MD5 `b7b30f96d466fba86016aa5a26606e0f`) giải nén vào `data/images/`
ở **gốc repo**; `labels.csv`, `{train,val,test}_subset0.csv` từ github.com/AlexOlsen/DeepWeeds vào `data/labels/`.
Không sửa, không lọc, không chia lại (S1). Dữ liệu và checkpoint **không** commit (`data/`, `runs/` nằm trong `.gitignore`).

## Thứ tự chạy (từ thư mục bài nộp `submissions/2A202602515_DoanDuyBach/`)

Đây đúng là các lệnh đã chạy để ra số liệu trong bài (phần sau Bước 2 được tự động hoá bằng `code/run_queue.sh` và
`code/post_queue.sh`; hai script này bỏ qua bước đã xong và tiếp tục run bị ngắt).

```bash
cd code && python -m unittest test_code && cd ..                 # 25 unit test (focal γ=0 ≡ CE, CutMix, gộp BN, ...)
python code/step0_eda_sanity.py                                    # EDA + kiểm tra split + loss ban đầu + overfit 1 batch
python code/experiments.py B01 B02 B03 B04 B05 B06 D01             # Bước 1: 6 backbone + 1 chẩn đoán (seed 0)
python code/bench_train.py                                         # tốc độ train đo cùng điều kiện
python code/experiments.py --alias B03 T00                         # T00 seed 0 = B03 (cùng cấu hình)
python code/experiments.py T01 T02 T03 T04 T05 T06 T07 T08 T09     # Bước 2: ablation + kết hợp trên DeiT-S
python code/step3_inference.py --exp T00 --backbone deit_small_patch16_224 --ema-exp T08 --ensemble B03 B04 B02
python code/experiments.py T00 --seeds 1 2                         # mốc: thêm 2 seed
python code/experiments.py --alias T09 F01                         # F01 seed 0 = T09 (cùng cấu hình)
python code/experiments.py F01 --seeds 1 2                         # chung kết: thêm 2 seed
python code/diag_precise_bn.py                                     # chẩn đoán BN của CNN Bước 1 (chỉ val)
# Bước 4: TEST đúng một lần mỗi seed. Mốc = công thức nền + 1 view (I00);
# chung kết = ảnh đầy đủ resize 288, 1 view (I04_288) + temperature scaling khớp trên val từng seed
python code/step4_final.py --exp T00 --backbone deit_small_patch16_224 --seeds 0 1 2 --views none --eval-mode crop --img-size 224
python code/step4_final.py --exp F01 --backbone deit_small_patch16_224 --seeds 0 1 2 --views none --eval-mode full --img-size 288 --temperature
L=../../data/labels
for tag in F01 T00 F01uncal; do
    python ../../eval.py score --pred "predictions/${tag}_seed*_test.csv" --test-csv $L/test_subset0.csv \
        --labels $L/labels.csv --tag $tag --out eval_out
done
P95=$(python -c "import json;print(json.load(open('logs/step3_latency.json'))['I04_res288_b1']['p95'])")
python ../../eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" \
    --val-csv $L/val_subset0.csv --latency-p95-ms $P95 --latency-method proper \
    --test-csv $L/test_subset0.csv --labels $L/labels.csv --out eval_out
mkdir -p logs/eval && cp eval_out/* logs/eval/                     # eval_out/ bị .gitignore; bản nộp ở logs/eval/
python code/diag_factorial_val.py                                  # phân tích sau chung kết, CHỈ trên val
python code/make_results.py && python code/make_figures.py         # results.xlsx + figures/ + curves T00/F01 (3 seed)
```

Mọi thí nghiệm huấn luyện đi qua **một** hàm `train.run(Config)`; danh sách cấu hình ở `code/experiments.py`.
Mỗi lần chạy có checkpoint mỗi epoch (`runs/<exp>/seed<k>/last.pt`), chạy lại cùng lệnh sẽ tiếp tục từ epoch kế tiếp.
`step4_final.py` có file khoá `runs/<exp>/seed<k>/test_done.json`: test chỉ chạy **một lần mỗi seed**.

### Phần làm thêm (điểm thưởng, report mục 10)

Chạy sau phần chính, từ thư mục bài nộp. Không script nào dùng test fold 0; nhiều fold dùng đủ bộ ba file của fold đó
(S6), test của mỗi fold chạy một lần.

```bash
pip install onnx onnxruntime onnxscript                            # chỉ cần cho bonus_onnx.py (uv: uv pip install ...)
bash code/run_folds.sh >> logs/folds.out 2>&1                      # F01 trên fold 1, 2 (seed 0) + test 1 lần/fold + eval.py
python code/bonus_dinov2_probe.py                                  # linear probe DINOv2 / DeiT-S đóng băng (train -> val)
python code/bonus_shift_tta.py shift                               # lệch phân phối trên val: macro-F1, ECE trước/sau TS
python code/bonus_shift_tta.py bn                                  # TTA: chuẩn hoá lại thống kê BN (B05, B06)
python code/bonus_shift_tta.py tent                                # TTA: Tent cho F01 (DeiT-S, LayerNorm)
python code/bonus_attention.py                                     # Grad-CAM + attention rollout, lỗi Chinee ↔ Snake
python code/bonus_onnx.py                                          # xuất ONNX, so độ trễ với PyTorch (chạy khi máy rảnh)
python code/bench_amp_dtype.py                                     # tốc độ train ResNet-50 theo dtype (căn cứ chọn AMP)
python code/make_results.py && python code/make_figures.py         # thêm các sheet Bonus_* và AMP_dtype, hình bonus_*
```

**Thời gian chạy thực tế** (Apple M5, MPS): DeiT-S thường mất 30–50 phút cho một run 10 epoch; tổng thời gian train
của 20 run (không tính 2 bản sao `--alias`) khoảng 12 giờ. Trên GPU CUDA (T4 trở lên), cùng code chạy nhanh hơn nhiều.

## Cấu trúc

| Thư mục/file | Nội dung |
|---|---|
| `code/` | `dataset.py`, `model.py`, `losses.py`, `train.py`, `inference.py`, `benchmark.py` (khung `starter/` đã hoàn thiện), `experiments.py` (danh sách thí nghiệm + runner), `step0_eda_sanity.py`, `step3_inference.py`, `step4_final.py`, `make_results.py`, `make_figures.py`, `bench_train.py`, `diag_bn_mps.py`, `diag_precise_bn.py`, `diag_factorial_val.py`, `test_code.py`, `run_queue.sh`/`post_queue.sh`/`watchdog.sh` (hàng đợi chạy tự động), `lab_day2.ipynb` |
| `logs/` | Log đầy đủ từng lần chạy (`<exp>_seed<k>.log`), `config.json`/`history.csv`/`summary.json` của từng run, JSON của Bước 0, 3, 4 và của các chẩn đoán |
| `curves/` | Một ảnh biểu đồ training cho mỗi `exp_id` (loss train/val, macro-F1/top-1 val, LR theo bước); `T00` và `F01` vẽ đủ 3 seed |
| `predictions/` | Dự đoán val của mọi run; dự đoán **test** của chung kết `F01` và mốc `T00` (3 seed), `F01uncal_*` (chưa temperature scaling) |
| `figures/` | EDA, kiểm tra pipeline, biểu đồ phân tích |
| `logs/eval/` | Đầu ra của `eval.py score` / `grade` (bản sao của `eval_out/`, thư mục này nằm trong `.gitignore`) |

Checkpoint (`runs/`, ~90 MB mỗi run) không commit.
