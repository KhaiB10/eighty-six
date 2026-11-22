#!/usr/bin/env bash
# ncm14c.sh (runs ON the Dell): eighty-six CONTROL for the final test. In ncm14 every LM1B arm, even the random placebo,
# had ~6-9 points worse genPPL than the earlier "normal" runs (which ran WITHOUT the patch installed). Is that the locks,
# or my pipeline? Small cook only, ~10 min:
#   normal_again  = unpatched, nothing locked (how much does the baseline move run to run? Phase 1 gave 93.15)
#   patched_empty = patch installed + NCM_MASK set, but every shelf's lock list is EMPTY (the pipeline's own effect)
#   rand_midfine  = random lock on shelves 4-11 only (is the first-shelf part of the random lock what hurts?)
set -uo pipefail
OUT=/data/lab/ncm14c; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
cd /data/ncm/ncm; PY=/data/lab/venv_ncm/bin/python
echo "=== eighty-six CONTROL: is the LM1B quality drop the locks or the pipeline? $(date) ==="
OLLAMA_STOPPED=0
if systemctl is-active --quiet ollama 2>/dev/null; then
  if sudo -n systemctl stop ollama 2>/dev/null; then OLLAMA_STOPPED=1; echo "   paused Ollama (Aster) for this run; it restarts automatically at the end"
  else echo "   !! could not pause Ollama without a password; run 'sudo systemctl stop ollama' first or Aster may crowd the GPU"; fi
fi
trap '[ $OLLAMA_STOPPED = 1 ] && sudo -n systemctl start ollama && echo "   Ollama (Aster) restarted"' EXIT
for m in $(curl -s -m 5 http://localhost:11434/api/ps 2>/dev/null | grep -o '"name":"[^"]*"' | cut -d'"' -f4); do
  curl -s -m 20 http://localhost:11434/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" >/dev/null && echo "   unloaded $m"
done
sleep 3; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | sed 's/^/   GPU before start: /'




[ -f /data/lab/ncm14/rand_midfine.pt ] || { echo "!! need ncm14 masks"; exit 1; }
$PY -c "
import torch; m = torch.load('/data/lab/ncm14/rand_midfine.pt'); torch.save({l: v[:0] for l, v in m.items()}, '/data/lab/ncm14c/empty.pt'); print('   empty mask written')"
echo; echo "-- arm: normal_again (unpatched)"
timeout 3600 $PY evaluate.py --config config/eval/lm1b-small.yaml --checkpoint-dir artifacts/lm1b/small --out $OUT/normal_again 2>&1 | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -7
echo "-- 3. patch the cook (v2: also locks the first shelf) (guarded; inactive without NCM_MASK)"
cp scripts/eval_shard.py $OUT/eval_shard.py.orig
if ! grep -q "NCM_MASK_PATCH" scripts/eval_shard.py; then
  ln=$(grep -n "^net = NextConceptTransformer(" scripts/eval_shard.py | head -1 | cut -d: -f1)
  [ -z "$ln" ] && { echo "!! could not find where the cook is built"; exit 1; }
  head -n $((ln-1)) scripts/eval_shard.py > /tmp/es.py
  cat >> /tmp/es.py <<'PATCH'
# NCM_MASK_PATCH (Khai's ncm3 cook test): forbid masked jars at each level while sampling. Inactive without NCM_MASK.
import os as _os
if _os.environ.get("NCM_MASK"):
    import torch as _t
    from models.next_concept import NextConceptTransformer as _NCT
    _MASK = {int(k): v.long() for k, v in _t.load(_os.environ["NCM_MASK"]).items()}
    _HITS = {"l0_masked": 0, "l0_total": 0}
    _og, _os_ = _NCT.generate, _NCT._sample_level
    def _gen(self, idx, *a, **k):
        if idx is not None and 0 in _MASK:
            _HITS["l0_total"] += idx.numel(); _HITS["l0_masked"] += _t.isin(idx, _MASK[0].to(idx.device)).sum().item()
            # v2 fix: the first shelf is drawn by eval_shard's unigram sampler (L0_P), outside the cook. Redraw any locked jar.
            if len(_MASK[0]):
                import sys as _s; _P0 = getattr(_s.modules["__main__"], "L0_P", None)
                _bad = _t.isin(idx, _MASK[0].to(idx.device))
                if _P0 is not None and _bad.any():
                    _p = _P0.clone().to(idx.device).float(); _p[_MASK[0].to(idx.device)] = 0; _p = _p / _p.sum()
                    idx = idx.clone(); idx[_bad] = _t.multinomial(_p, int(_bad.sum()), replacement=True)
        self._lvl = 1 if idx is not None else 0
        return _og(self, idx, *a, **k)
    def _samp(self, logits, *a, **k):
        l = getattr(self, "_lvl", 0); self._lvl = l + 1; c = _MASK.get(l)
        if c is not None and len(c):
            logits = logits.clone(); logits[..., c.to(logits.device)] = float("-inf")
        return _os_(self, logits, *a, **k)
    _NCT.generate, _NCT._sample_level = _gen, _samp
    import atexit; atexit.register(lambda: print("NCM_MASK level-0 picks that were masked jars:", _HITS))
PATCH
  tail -n +$ln scripts/eval_shard.py >> /tmp/es.py; cp /tmp/es.py scripts/eval_shard.py
fi
$PY -m py_compile scripts/eval_shard.py && echo "   patched ok"





for a in patched_empty:empty rand_midfine:/data/lab/ncm14/rand_midfine.pt; do
  n=${a%%:*}; m=${a#*:}; [ "$m" = empty ] && m=$OUT/empty.pt
  echo; echo "-- arm: $n"
  NCM_MASK=$m timeout 3600 $PY evaluate.py --config config/eval/lm1b-small.yaml --checkpoint-dir artifacts/lm1b/small --out $OUT/$n 2>&1 | grep -vE "[Ww]arn|deprecat|Loading weights|it/s\]" | tail -7
done
cp $OUT/eval_shard.py.orig scripts/eval_shard.py && echo && echo "-- eval_shard.py restored to original"
echo; echo "How to read: normal_again ~93 and patched_empty ~93 -> the pipeline is fine and locking ~700 LM1B jars really costs"
echo "quality. patched_empty ~100 -> my pipeline causes it (bug to fix before publishing). Compare rand_midfine with ncm14's"
echo "small_rand_all (103.90) to see whether the first shelves are the sensitive part."
echo "=== DONE. Tell Claude: ncm14c ready ==="
