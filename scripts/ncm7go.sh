#!/usr/bin/env bash
# ncm7go.sh: start ncm7.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm7
nohup bash /data/lab/ncm7.sh > /data/lab/ncm7/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm7/REPORT.txt; pgrep -af ncm7.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm7/nohup.log"
