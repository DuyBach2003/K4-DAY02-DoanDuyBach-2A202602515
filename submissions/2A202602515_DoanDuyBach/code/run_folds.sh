#!/bin/bash
# Điểm thưởng (RUBRIC mục 2): cấu hình chung kết F01 trên fold 1 và 2 của tác giả (seed 0), mỗi fold dùng đủ bộ ba file
# của chính fold đó (S6); T khớp trên val của fold đó; test chạy MỘT lần mỗi fold. Fold 0 là bài chính (F01 seed 0).
# Chạy từ thư mục bài nộp:  caffeinate -ims bash code/run_folds.sh >> logs/folds.out 2>&1
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
export PYTHONWARNINGS=ignore HF_HUB_DISABLE_XET=1
L=../../data/labels
echo "=== folds start $(date)"
# watchdog: experiments.py không ghi dòng log mới > 15 phút (1 epoch ~3-5 phút) thì coi là treo và kill;
# vòng attempt bên dưới tiếp tục từ runs/<exp>/seed0/last.pt
( while sleep 60; do
    pgrep -f "code/experiments.py F01_fold" >/dev/null || continue
    if [ $(( $(date +%s) - $(stat -f %m logs/folds.out) )) -gt 900 ]; then
        echo "$(date) watchdog: không có log mới > 15 phút, kill experiments.py"
        pkill -f "code/experiments.py F01_fold"
    fi
  done ) &
WD=$!
for attempt in 1 2 3; do
    $PY code/experiments.py F01_fold1 F01_fold2
done
kill $WD
for k in 1 2; do
    [ -f runs/F01_fold$k/seed0/summary.json ] || { echo "THIẾU runs/F01_fold$k/seed0/summary.json, dừng"; exit 1; }
    $PY code/step4_final.py --exp F01_fold$k --fold $k --backbone deit_small_patch16_224 --seeds 0 --views none \
        --eval-mode full --img-size 288 --temperature
    $PY ../../eval.py score --pred "predictions/F01_fold${k}_seed*_test.csv" --test-csv $L/test_subset$k.csv \
        --labels $L/labels.csv --tag F01_fold$k --out eval_out
done
mkdir -p logs/eval && cp eval_out/* logs/eval/
echo "=== folds end $(date)"
