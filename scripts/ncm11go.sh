#!/usr/bin/env bash
# ncm11go.sh: start ncm11.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm11
nohup bash /data/lab/ncm11.sh > /data/lab/ncm11/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm11/REPORT.txt; pgrep -af ncm11.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm11/nohup.log"
