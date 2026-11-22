#!/usr/bin/env bash
# ncm9go.sh: start ncm9.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm9
nohup bash /data/lab/ncm9.sh > /data/lab/ncm9/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm9/REPORT.txt; pgrep -af ncm9.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm9/nohup.log"
