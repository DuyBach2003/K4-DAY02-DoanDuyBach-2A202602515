#!/bin/bash
# Hàng đợi chạy phần còn lại của bài lab, tuần tự trên 1 GPU. Idempotent: bước đã xong thì bỏ qua,
# run bị ngắt thì tiếp tục từ runs/<exp>/seed<k>/last.pt. Chạy từ thư mục bài nộp:
#     caffeinate -ims bash code/run_queue.sh >> logs/queue.out 2>&1
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
export PYTHONWARNINGS=ignore HF_HUB_DISABLE_XET=1

echo "=== queue start $(date)"
$PY code/experiments.py T06 T07 T04 T01                               # Bước 2: phần ablation còn lại
[ -f logs/bench_train.json ] || $PY code/bench_train.py               # tốc độ train cùng điều kiện
[ -f logs/step3_inference.json ] || $PY code/step3_inference.py --exp T00 --backbone deit_small_patch16_224 \
    --ema-exp T08 --ensemble B03 B04 B02                              # Bước 3
$PY code/experiments.py T00 --seeds 1 2                               # mốc chung kết: thêm 2 seed
# Bước 2 kết hợp + chung kết (cấu hình chốt trong experiments.py sau khi xem kết quả ablation)
if [ -f code/queue_final.flag ]; then
    $PY code/experiments.py T09
    [ -d runs/F01/seed0 ] || $PY code/experiments.py --alias T09 F01
    $PY code/experiments.py F01 --seeds 1 2
fi
echo "=== queue end $(date)"
