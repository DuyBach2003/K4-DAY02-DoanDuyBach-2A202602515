"""Chạy một lệnh tách hẳn khỏi terminal/phiên hiện tại (session mới, stdin/stdout chuyển hướng vào file log).

    python code/launch_detached.py logs/runner.out python code/experiments.py T01 T02 ...
"""
import os
import sys

log, cmd = sys.argv[1], sys.argv[2:]
if os.fork() > 0:
    sys.exit(0)
os.setsid()
if os.fork() > 0:
    sys.exit(0)
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
