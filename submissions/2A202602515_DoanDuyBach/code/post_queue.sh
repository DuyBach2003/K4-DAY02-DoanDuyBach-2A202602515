#!/bin/bash
# Phần cuối sau khi train xong: chẩn đoán precise-BN, TEST một lần mỗi seed (T00 mốc, F01 chung kết),
# eval.py score/grade, results.xlsx, biểu đồ. Chạy từ thư mục bài nộp: bash code/post_queue.sh >> logs/post.out 2>&1
set -u
cd "$(dirname "$0")/.."
PY=../../.venv/bin/python
export PYTHONWARNINGS=ignore
L=../../data/labels
BB=deit_small_patch16_224
echo "=== post start $(date)"
# hoàn tất các run bị ngắt (idempotent: run đã có summary.json thì bỏ qua, run dở thì resume từ last.pt)
for attempt in 1 2 3; do
    $PY code/experiments.py T00 --seeds 1 2
    $PY code/experiments.py T09
    [ -d runs/F01/seed0 ] || $PY code/experiments.py --alias T09 F01
    $PY code/experiments.py F01 --seeds 1 2
done
for s in 0 1 2; do
    for e in T00 F01; do
        [ -f runs/$e/seed$s/summary.json ] || { echo "THIẾU runs/$e/seed$s/summary.json, dừng"; exit 1; }
    done
done
[ -f logs/diag_precise_bn.json ] || $PY code/diag_precise_bn.py
# mốc: công thức nền + 1 view (I00); chung kết: ảnh đầy đủ 288, 1 view (I04_288) + temperature khớp trên val
$PY code/step4_final.py --exp T00 --backbone $BB --seeds 0 1 2 --views none --eval-mode crop --img-size 224
$PY code/step4_final.py --exp F01 --backbone $BB --seeds 0 1 2 --views none --eval-mode full --img-size 288 --temperature
for tag in F01 T00 F01uncal; do
    $PY ../../eval.py score --pred "predictions/${tag}_seed*_test.csv" --test-csv $L/test_subset0.csv \
        --labels $L/labels.csv --tag $tag --out eval_out
done
P95=$($PY -c "import json;print(json.load(open('logs/step3_latency.json'))['I04_res288_b1']['p95'])")
$PY ../../eval.py grade --final "predictions/F01_seed*_test.csv" --baseline "predictions/T00_seed*_test.csv" \
    --uncal "predictions/F01uncal_seed*_test.csv" --final-val "predictions/F01_seed*_val.csv" \
    --val-csv $L/val_subset0.csv --latency-p95-ms $P95 --latency-method proper \
    --test-csv $L/test_subset0.csv --labels $L/labels.csv --out eval_out
mkdir -p logs/eval && cp eval_out/* logs/eval/
$PY code/make_results.py
$PY code/make_figures.py
echo "=== post end $(date)"
