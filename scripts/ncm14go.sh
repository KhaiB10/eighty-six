#!/usr/bin/env bash
# ncm14go.sh: start ncm14.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm14
nohup bash /data/lab/ncm14.sh > /data/lab/ncm14/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm14/REPORT.txt; pgrep -af ncm14.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm14/nohup.log"
