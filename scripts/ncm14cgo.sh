#!/usr/bin/env bash
# ncm14cgo.sh: start ncm14c.sh in the background so closing ssh doesn't kill it.
mkdir -p /data/lab/ncm14c
nohup bash /data/lab/ncm14c.sh > /data/lab/ncm14c/nohup.log 2>&1 < /dev/null &
sleep 20; echo "--- first lines:"; head -12 /data/lab/ncm14c/REPORT.txt; pgrep -af ncm14c.sh >/dev/null && echo "RUNNING (safe to disconnect)" || echo "NOT RUNNING - see /data/lab/ncm14c/nohup.log"
