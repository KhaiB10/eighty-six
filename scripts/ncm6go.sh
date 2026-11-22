#!/usr/bin/env bash
# ncm6go.sh (runs ON the Dell): start ncm6.sh DETACHED, so closing ssh / the phone does not kill it.
# Watch progress any time with:  tail -f /data/lab/ncm6/REPORT.txt   (Ctrl+C stops watching, not the run)
mkdir -p /data/lab/ncm6
nohup setsid bash /data/lab/ncm6.sh > /data/lab/ncm6/nohup.log 2>&1 < /dev/null &
echo "started ncm6 in the background (pid $!). Safe to disconnect."
echo "check progress:  ssh dell 'tail -5 /data/lab/ncm6/REPORT.txt'"
