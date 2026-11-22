#!/usr/bin/env bash
# ncm5b.sh (runs ON the Dell): eighty-six PHASE 5B, "the taste tester". Word lists can't see a beach story that avoids
# beach words ("surfboard", "castles", "the big whale"). Here a local LLM judge (Aster's Qwen3:8b via Ollama) reads each
# story and answers YES/NO: is this story set at / about a beach or the sea? Same 400 story slots per arm, 5 arms.
# Key number: HIDDEN = judge says beach, word list says no. ~15-30 min (prints its speed early). Read-only.
set -uo pipefail
OUT=/data/lab/ncm5b; mkdir -p $OUT; exec > >(tee $OUT/REPORT.txt) 2>&1
echo "=== eighty-six PHASE 5B: LLM TASTE TESTER $(date) ==="
MODEL=${JUDGE_MODEL:-qwen3:8b}
curl -s -m 5 http://localhost:11434/api/tags | grep -q "\"$MODEL\"" || { echo "!! Ollama at localhost:11434 has no model $MODEL"; curl -s -m 5 http://localhost:11434/api/tags | head -c 600; exit 1; }
echo "-- judge: $MODEL"
MODEL=$MODEL /data/lab/venv_ncm/bin/python - <<'P'
import glob, pickle, re, json, random, time, os, urllib.request
MODEL = os.environ["MODEL"]; random.seed(0)
BEACH = r"\b(beach|beaches|sand|sandy|sandcastle|sandcastles|ocean|sea|seaside|shore|shell|shells|seashell|seashells|wave|waves|surf|tide)\b"
ARMS = {"none": "/data/lab/ncm3/none", "ban": "/data/lab/ncm5/ban", "coarse32": "/data/lab/ncm5/coarse32",
        "middle32": "/data/lab/ncm5/middle32", "all32": "/data/lab/ncm5/all32"}
H = {a: [h for p in sorted(glob.glob(f"{d}/shard_*.pkl")) for h in pickle.load(open(p, "rb"))["hyps"]] for a, d in ARMS.items()}
n = min(len(v) for v in H.values())
# all slots where ANY arm's word list fires (so every beach story is judged) + random others, 400 total
hot = sorted({i for a in H for i in range(n) if re.search(BEACH, H[a][i], re.I)})
rest = [i for i in range(n) if i not in set(hot)]; random.shuffle(rest)
idx = sorted(hot + rest[:max(0, 400 - len(hot))])
print(f"   slots judged per arm: {len(idx)} ({len(hot)} where some arm used beach words + {len(idx)-len(hot)} random)")
Q = ("Read this short children's story. Is any part of it set at a beach, the seaside or the ocean, or about sand, "
     "sea waves, seashells or surfing? Answer with only YES or NO.\n\nStory:\n{s}\n\nAnswer (YES or NO):")
def ask(story):
    body = {"model": MODEL, "prompt": Q.format(s=story[:1200]), "stream": False, "think": False,
            "options": {"temperature": 0, "num_predict": 6}}
    for attempt in range(2):
        try:
            req = urllib.request.Request("http://localhost:11434/api/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=120).read())["response"]
            r = re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip().upper()
            return r.startswith("Y") if r[:1] in "YN" else ("YES" in r)
        except Exception as e:
            if attempt == 0 and "think" in body: body.pop("think"); body["prompt"] = "/no_think " + body["prompt"]; continue
            print("   judge error:", type(e).__name__, str(e)[:150]); return None
R, t0, done = {}, time.time(), 0
for a in ARMS:
    R[a] = []
    for i in idx:
        R[a].append((i, ask(H[a][i]), bool(re.search(BEACH, H[a][i], re.I)))); done += 1
        if done == 25: print(f"   speed: {(time.time()-t0)/25:.2f}s per story -> about {(time.time()-t0)/25*len(idx)*len(ARMS)/60:.0f} min total")
json.dump(R, open("/data/lab/ncm5b/judgements.json", "w"))
# judge sanity check on the normal cook: agreement with the word list
nz = [(j, w) for _, j, w in R["none"] if j is not None]
tp = sum(j and w for j, w in nz); fn = sum((not j) and w for j, w in nz); fp = sum(j and not w for j, w in nz)
print(f"\n   judge sanity (normal cook): of {tp+fn} word-list beach stories the judge caught {tp}; it also flagged {fp} stories with no beach words")
print("\n   arm      | judge: beach | word list: beach | HIDDEN beach (judge yes, words no) | judge errors")
for a, rows in R.items():
    ok = [(j, w) for _, j, w in rows if j is not None]; m = len(ok)
    print(f"   {a:<8} | {sum(j for j,_ in ok)/m*100:6.1f}%      | {sum(w for _,w in ok)/m*100:6.1f}%          | "
          f"{sum(j and not w for j,w in ok)/m*100:6.1f}%                             | {len(rows)-m}")
print("   (percentages are of the judged slots, which over-sample beach stories; compare arms to each other, not to Phase 5's 4.9%)")
for a in ("ban", "middle32", "all32"):
    hid = [i for i, j, w in R[a] if j and not w][:2]
    for i in hid: print(f"   hidden beach in [{a}]: {H[a][i][:170].replace(chr(10),' ')}")
P
echo "=== DONE. Tell Claude: ncm5b ready ==="
