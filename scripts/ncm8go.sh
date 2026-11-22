#!/usr/bin/env bash
# ncm8go.sh: start ncm8.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm8
nohup bash /data/lab/ncm8.sh > /data/lab/ncm8/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm8/REPORT.txt; pgrep -af ncm8.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm8/nohup.log"
