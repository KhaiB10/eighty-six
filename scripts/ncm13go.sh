#!/usr/bin/env bash
# ncm13go.sh: start ncm13.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm13
nohup bash /data/lab/ncm13.sh > /data/lab/ncm13/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm13/REPORT.txt; pgrep -af ncm13.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm13/nohup.log"
