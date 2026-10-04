#!/bin/bash
# Watchdog: nếu experiments.py đang chạy mà log (queue.out/post.out) không có dòng mới > 15 phút (1 epoch ~4-5 phút)
# thì coi là treo (DataLoader worker bị kill do thiếu RAM làm process chính deadlock) và kill; post_queue.sh sẽ resume.
cd "$(dirname "$0")/.."
while ! grep -q "=== post end" logs/post.out 2>/dev/null; do
    sleep 60
    if pgrep -f "code/experiments.py" >/dev/null; then
        newest=$(stat -f %m logs/queue.out logs/post.out 2>/dev/null | sort -n | tail -1)
        if [ $(( $(date +%s) - newest )) -gt 900 ]; then
            echo "$(date) watchdog: không có log mới > 15 phút, kill experiments.py" | tee -a logs/post.out
            pkill -f "code/experiments.py"
        fi
    fi
done
