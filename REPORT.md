# eighty-six: taking a concept off the menu of a Next-Concept Model
### Phases 2–7: where names and topics live in NCM's code hierarchy, locking them out at generation time, and whether the lock scales

**Author:** Khai Bustos (GitHub: KhaiB10) · **Date:** 2026-11-20
**Model:** Next-Concept Modeling (NCM), arXiv:2610.14162 (Hazy Research / Together AI), released TinyStories checkpoint
**Code:** github.com/HazyResearch/ncm @ c702742 (MIT) · **Hardware:** one Tesla P40 (2016), plain FP32
**Phase 1** (replication of the released checkpoints on the P40): see `PHASE1_REPORT.md`.

> *To "86" something, in kitchen slang, is to take it off the menu.*

## TL;DR
- NCM's tokenizer writes every text as 12 levels of discrete codes (coarse → fine), each level fixing what the levels above got wrong. Its generator writes level by level, all positions of a level at once.
- **A generation-time code lock:** audit which codes a target uses on real examples, then forbid the generator from picking them. No retraining, no extra forward passes, and the lock is a file you can inspect.
- **Names:** locking ~500 codes (1% of all codes) on the middle + fine levels cut stories mentioning "Lily" from **14.9% → 1.3%**, with girl heroes mostly kept under other names and a small fluency cost (genPPL 16.73 vs 16.39).
- **Topics:** locking 384 codes on all levels removed beach stories (LLM judge: 12.0% → 0.0%) and dog stories (17.4% → 0.5%) with no fluency cost and unrelated content (park, girls, she/her) unchanged, though some general water/animal stories went with them. A **word ban** only hid the words: the judge still found beach stories in 5.2% ("pretty seaweed", "the deep seas, the reef").
- **Controls:** usage-matched random locks of the same size changed nothing on TinyStories, so the effects are specific. Each lock only hit its own category.
- **Where things live:** what kind of story → coarse levels; which name → middle levels, *unless* the name shares codes with its kind (dog names do, girl names mostly don't), in which case locking it removes the kind too. Two predictions made before running (Max, Sue) held in direction; Sue's girl-loss was underpredicted.
- **Scale (real news, LM1B):** the same audited lock removed "obama" at 97M, 317M and 725M generator sizes (≈15% → 0.2%); the larger generators did not route around it. It also removed "president" and other politicians, so on news it acted as a topic lock. On real news, any lock of ~500–750 codes, random ones included, cost ~7–10% genPPL; on TinyStories it cost nothing.
- **Remembering is harder:** teaching a new name by training only per-level output heads failed; a full fine-tune learned it but over-used it (76% of stories) and doubled genPPL.

## Method
1. **Audit (tokenizer only).** Encode 1,500 TinyStories validation stories containing "Lily" and 4,000 without. For each level, map each Lily token position to the code covering it, and score codes by log-odds (Lily positions vs. all background positions), the same idea as the TF-IDF slot ranking in sparse memory finetuning (arXiv:2510.15103).
2. **Mask (tokenizer only).** Set the masked codebook rows to 1e4 so nearest-code assignment can never pick them; re-encode held-out stories (500 Lily, 1,000 other, 400 "Tim" control) and decode. Masked-code leaks were 0 in every run.
3. **Iterate.** Re-audit after masking and add the next top-32 codes per level, for up to 8 rounds.
4. **Lock the generator.** A guarded patch in `scripts/eval_shard.py` sets the logits of masked codes to −inf before each level is sampled (level 0 is drawn by the eval's unigram sampler and cannot be masked; 26/1,280 level-0 draws hit a masked code in the r2 arm). Then run the paper's official TinyStories evaluation (1,280 samples, 5 seeds, genPPL scored by GPT-2 large).
5. **Baseline: word ban.** Forbid the Lily token ids (413, 674) in the tokenizer decoder's final output, the NCM equivalent of a normal LLM bad-words list.
6. **Counting.** One count per generated story (`hyps` field). "Girl hero" = story mentions any of 14 girl names; "she/her" = story uses she or her.

## Results

### A. Tokenizer audit: Lily uses specialty codes on every level
| Level | Positions | Distinct codes under Lily | Top-8 / top-32 codes cover | Top-32 codes' share of background use |
|---|---|---|---|---|
| L0 | 1 | 1,025 | 1.9 / 5.9 % | 0.00 % |
| L3 | 24 | 1,559 | 20.7 / 38.1 % | 0.98 % |
| L4 | 32 | 1,102 | 41.0 / 60.5 % | 1.17 % |
| L7 | 64 | 316 | 66.5 / 81.4 % | 1.03 % |
| L11 | 128 | 51 | **97.5** / 99.5 % | 1.31 % |

Baseline reconstruction is 99.99% token-accurate; Lily is reconstructed in 100% of stories.

### B. Tokenizer masking: the residual hierarchy self-heals
| Codes masked | Lily removed from reconstruction | Accuracy on other stories |
|---|---|---|
| Any single level group (coarse / middle / fine), up to 128 per level | 0–0.2% | unchanged |
| All levels, 32 per level (round 1, 376 codes) | 40.0% | 99.99% |
| Round 2 (760 codes, 1.5%) | 62.8% | 99.98% |
| Round 8 (3,064 codes, 6.2%) | 79.4% | 99.20% |

Masking one group of levels fails because later levels correct the residual and put Lily back. Replacements degrade in order: other girls' names (rounds 1–2) → boys' names (round 3+) → animals, e.g. "Fluffy", "Kitty" (round 3+).

### C. Generator lock vs. word ban (official eval, 1,280 stories, 5 seeds)
| Arm | Codes locked | Lily | Girl hero | she/her | Tim | genPPL (± seeds) |
|---|---|---|---|---|---|---|
| Normal | 0 | 14.9% | 42.3% | 47.5% | 28.1% | 16.39 ± 0.33 |
| Word ban | — | 0.0% | 40.9% | 47.5% | 28.6% | 17.14 ± 0.31 |
| All levels, r2 | 760 | 0.0% | 26.9% | 40.0% | 37.6% | 16.47 ± 0.34 |
| All levels, r8 | 3,064 | 0.0% | ~7% | — | 29.4% | 17.81 ± 0.43 |
| Coarse only L0–3 | 256 | 4.7% | 28.9% | 40.5% | 36.6% | 16.44 ± 0.23 |
| Middle only L4–7 | 256 | 3.0% | 39.1% | 47.3% | 28.7% | 16.56 ± 0.27 |
| Fine only L8–11 | 248 | 14.9% | 42.4% | 47.7% | 28.7% | 16.55 ± 0.28 |
| **Middle + fine L4–11** | **504** | **1.3%** | **38.3%** | **47.2%** | 29.1% | 16.73 ± 0.22 |

The word ban was worse than normal on all 5 seeds (+0.63 to +0.93 genPPL). Arms are sampled independently after the lock, so seeds are not perfectly paired.

**Same sample slot, three cooks:**
- Normal: *"…a little girl named Lily. She had a toy bear named Ted…"*
- Word ban: *"…a little girl named.. She had a toy bear named Ted… .. was always dependable."*
- Middle + fine lock: *"…a little girl named Mia. She had a toy bear named Ted… Mia was always…"*


### D. Phase 4: tuning the middle lock hits a floor
Audit restricted to the middle levels (L4–7), 32 new codes per level per round, with and without the fine-level codes from r2.

| Arm | Codes | Lily | Girl hero | she/her | genPPL |
|---|---|---|---|---|---|
| mid r1 | 128 | 3.0% | 39.3% | 47.6% | 16.51 ± 0.31 |
| mid r2 | 256 | 2.7% | 39.4% | 47.1% | 16.49 ± 0.32 |
| mid r4 | 512 | 3.3% | 37.0% | 46.2% | 16.65 ± 0.36 |
| midfine r1 | 376 | 1.4% | 38.7% | 47.7% | 16.68 ± 0.25 |
| **midfine r2** | **504** | **1.2%** | **38.8%** | **47.0%** | **16.66 ± 0.28** |
| midfine r3 | 632 | 1.2% | 36.8% | 46.2% | 16.69 ± 0.23 |
| midfine r4 | 760 | 1.8% | 36.2% | 46.2% | 16.79 ± 0.39 |

More rounds do not lower the ~1–3% Lily floor; they slowly erode girl-hero stories. One to two rounds is enough. The remaining leaks often come back **later in the story** ("a girl named Amy. Lily liked to draw…"), i.e. the generator reaches Lily through code combinations the audit did not flag. Level 0 (drawn by the eval's unigram sampler) is also unmasked.


### E. Phase 5: taking a whole topic (beach stories) off the menu
Audit at story level: 1,019 stories mentioning beach words vs 4,000 without; top codes per level by log-odds (top-32 beach codes are 26–700× more common in beach stories than in background on L1–L11; L0 only 1.1×). Word ban = all 16 single-token beach words (sand, sea, beach, shell, wave(s), shore, ocean, surf, tide, sandcastle(s), sandy…).

| Arm | Codes | Beach words | Water/swim/boat | Park | Girl hero | she/her | genPPL |
|---|---|---|---|---|---|---|---|
| Normal | 0 | 4.9% | 10.8% | 26.2% | 42.3% | 39.5% | 16.39 ± 0.33 |
| Word ban | — | 0.0% | 11.2% | 27.0% | 42.3% | 39.5% | 16.48 ± 0.29 |
| Coarse L0–3 ×32 | 128 | 1.3% | 7.5% | 27.0% | 42.9% | 40.9% | 16.32 ± 0.25 |
| **Middle L4–7 ×32** | **128** | **1.2%** | **10.7%** | **26.7%** | **42.2%** | **39.7%** | **16.36 ± 0.28** |
| Fine L8–11 ×32 | 128 | 3.4% | 11.1% | 26.6% | 42.3% | 39.5% | 16.47 ± 0.30 |
| All L0–11 ×32 | 384 | 0.0% | 7.0% | 27.3% | 42.9% | 40.9% | 16.33 ± 0.21 |
| Coarse L0–3 ×128 | 512 | 1.1% | 4.4% | 25.9% | 43.0% | 42.1% | 16.16 ± 0.28 |

**The word ban hides the words, not the topic.** Same sample slot:
- Normal: *"…went to the **beach** with her mom… Lily wanted to **surf**…"* → Ban: *"…went to the park… Lily wanted to **surfboard**…"*; *"play in the **sand**"* → Ban: *"play in the the and **castles**"*; *"near the big **sea**"* → Ban: *"near the big **whale**"*.
- All-shelf lock: *"…went to the big park… a big tree with a swing… fly on the kite"*; *"a little house **in the woods**"*: the scene itself changes.

Findings: (1) unlike a name, a topic is removed by locking **coarse or middle** levels; fine levels alone are weak (3.4%). (2) Coarse locks also cut general water stories (10.8% → 7.5% → 4.4%), middle locks do not (10.7%): coarse levels hold the broad scene ("near water"), middle levels the specific one ("beach"). (3) No lock hurt fluency. (4) Word-based counting flatters the word ban; a concept-level judge is needed. Caveat: only 63 normal stories mention the beach, so 1.2% vs 1.3% is not distinguishable.


### F. Phase 5b: an LLM judge confirms the word ban hides the topic
Qwen3:8b (local, via Ollama) read 400 story slots per arm (all 63 slots where any arm used beach words + 337 random) and answered YES/NO: is any part set at a beach or the sea? Judge sanity on the normal cook: caught 46 of 63 word-list beach stories (73%), flagged 2 others.

| Arm | Judge: beach | Word list: beach | Hidden (judge yes, no beach words) | Topic removed (judge, vs normal) |
|---|---|---|---|---|
| Normal | 12.0% | 15.8% | 0.5% | — |
| Word ban | 5.2% | 0.0% | **5.2%** | 57% |
| Middle L4–7 ×32 | 4.5% | 4.0% | 1.0% | 62% |
| Coarse L0–3 ×32 | 2.8% | 4.2% | 0.0% | 77% |
| **All L0–11 ×32** | **0.0%** | **0.0%** | **0.0%** | **100%** |

The word ban's hidden beach stories avoid every banned word: *"go to the park and find many pretty seaweed"*, *"a little fish named Fin… dive deep in the deep seas. The reef…"*. The all-level lock (384 codes, 0.8% of all codes) removed the topic completely by this judge, with no fluency cost (genPPL 16.33) and park / girl-hero / she-her stories unchanged; its cost is fewer general water stories (10.8% → 7.0%). The middle lock's "hidden" cases were boats and sailing, borderline for the judge. **For a topic, the trade-off is: coarse/all levels remove it fully but take some of the surrounding scene (water); middle levels keep the scene but leave more of the topic.**


### G. Phase 6: teaching a new name ("Khai") — training only the per-level output heads does not work
"Khai" never appears in TinyStories and splits into 3 BPE pieces (K·ha·i); the frozen tokenizer already rebuilds it in 95% of renamed stories, so the open question is the generator. NCM's generator is one shared transformer plus 11 output heads (one per level, L1–L11, each a 768→384 projection tied to that level's codebook). Training: 1,000 TinyStories validation stories with Lily → Khai, 150 steps, batch 32 (micro-batches of 8), AdamW (heads lr 1e-3, full 5e-5), FP32 on the P40. Generation: 512 unconditional samples, scored with GPT-2 large (own implementation; compare arms to each other).

| Arm | Trained params | Khai in new stories | Khai loss (held-out) | Normal-story loss | Girl hero | genPPL |
|---|---|---|---|---|---|---|
| None | 0 | 0.0% | 1.939 | 1.607 | 44.1% | 15.98 |
| Heads L1–3 (coarse) | 0.88M | 0.0% | 1.887 | 1.614 | 42.0% | 21.62 |
| Heads L4–7 (middle) | 1.18M | 0.0% | 1.818 | 1.553 | 41.2% | 17.48 |
| Heads L8–11 (fine) | 1.18M | 0.0% | 1.683 | 1.607 | 44.7% | 15.73 |
| All heads | 3.24M | 27.0% (often misspelled: "Ka", "Khaa") | 1.512 | 1.560 | 28.1% | 22.13 |
| Full fine-tune | 88.6M | **76.0%** | 1.214 | 1.285 | 20.7% | **29.93** |

Findings: (1) **No single head group can teach a new name**; the prediction that middle heads would suffice was wrong. A 3-piece name needs coordinated choices across levels (fine heads learned the spelling best by loss, with no harm, but never produced Khai). (2) All heads together partly learn it, misspelled, and degrade fluency. (3) The full fine-tune learns it but **over-learns**: Khai appears in 76% of stories (Lily was 15%), girl variety collapses, and genPPL nearly doubles. (4) Normal-story loss *fell* in most arms because the training stories come from the same distribution; it is not a usable forgetting measure here; genPPL and content variety are. **Remembering is harder than locking: a lock removes a choice, remembering has to create one at many levels at once.**


### H. Ladder step 1: a dog's name ("Spot") does not behave like a girl's name
Same recipe as Lily (2 audit rounds × 32 codes/level, then lock by level group). Spot: 1,058 validation stories (7,902 mentions); single tokens 601/1622. Pantry rounds: 43.8% → 70.9% removal; pantry replacements were other *animal* names (Tweet, Fluffy, Chirpy, Mittens, "Cat"), not other dog names.

| Lock | Codes | Spot | Dog stories (dog/puppy) | Other dog names | Tim | Lily | genPPL |
|---|---|---|---|---|---|---|---|
| None | 0 | 5.4% | 16.9% | 9.1% | 28.1% | 14.9% | 16.39 ± 0.33 |
| Coarse L0–3 | 256 | 0.6% | 8.0% | 3.4% | 30.8% | 14.7% | 16.39 ± 0.60 |
| Middle L4–7 | 256 | 2.4% | 11.8% | 7.9% | 28.7% | 14.8% | 16.50 ± 0.39 |
| Fine L8–11 | 233 | 4.9% | 16.0% | 9.0% | 29.1% | 15.5% | 16.59 ± 0.32 |
| Middle + fine | 489 | 2.2% | 10.3% | 7.3% | 29.8% | 15.2% | 16.71 ± 0.37 |
| All | 745 | 0.0% | 3.1% | 1.6% | 30.9% | 14.8% | 16.65 ± 0.59 |

Same slot: *"a charming little dog named Spot"* → coarse lock: *"a little boy named Tim went to the park"*; *"a cool little dog named Spot… sail on the big blue sea"* → middle lock: *"a cool little fish named Fin"*.

**The prediction (Spot behaves like Lily) was wrong.** Locking Spot's codes removes *dog stories*, not just the name: the coarse lock works best on Spot but halves dog stories; the all-level lock removes Spot and nearly all dogs. Name/girl controls (Tim, Lily) were unaffected; fluency was unchanged in all arms. *An earlier draft explained this as "Spot is fused with the dog". That explanation does not hold up: Spot appears in about 32% of dog stories and Lily in about 35% of girl stories, so neither name is more tied to its character type. The difference is real (confirmed by the placebo in section I) but not yet explained; section I's next step measures it directly.*


### I. Placebo: random locks of the same size change nothing; dog stories behave like the beach
Random codes matched to Lily's r2 codes level by level in **count and background usage** (median usage identical per level). The first-level leak is fixed in this run: codes drawn by the eval's level-0 sampler that are locked are redrawn (38/1,280 draws in arms that lock level 0). Dog stories counted without "bark". *(The original ncm8 scoreboard read the wrong folder; the dog-topic rows below are the corrected counts of ncm8's generations.)*

| Lock | Lily | Spot | Girl hero | Boy names | Dog stories | she/her | genPPL |
|---|---|---|---|---|---|---|---|
| Normal | 14.9% | 5.4% | 42.3% | 56.4% | 17.4% | 39.5% | 16.39 |
| **Random**, top L0–3 (256) | 14.8% | 4.3% | 43.2% | 55.9% | 17.0% | 44.1% | 16.45 |
| Lily, top L0–3 | 4.7% | 5.8% | 28.9% | 58.8% | 19.1% | 32.0% | 16.44 |
| Spot, top L0–3 | 14.7% | 0.6% | 42.5% | 56.7% | 8.1% | 38.9% | 16.39 |
| **Random**, middle L4–7 (256) | 15.2% | 5.1% | 42.1% | 56.2% | 18.0% | 40.5% | 16.40 |
| Lily, middle | 3.0% | 5.3% | 39.1% | 56.3% | 17.7% | 40.2% | 16.56 |
| Spot, middle | 14.8% | 2.4% | 42.4% | 57.0% | 12.4% | 40.4% | 16.50 |
| **Random**, fine L8–11 (248) | 14.9% | 5.4% | 42.4% | 56.3% | 17.3% | 39.3% | 16.56 |
| **Random**, all (760) | 14.5% | 4.1% | 42.8% | 56.0% | 17.3% | 43.0% | 16.77 |
| Lily, all (r2) | 0.0% | 5.7% | 26.9% | 58.7% | 19.6% | 32.0% | 16.47 |
| Spot, all | 14.8% | 0.0% | 42.5% | 56.1% | 3.4% | 38.8% | 16.65 |
| Dog topic, word ban | 14.9% | 5.4% | 42.3% | 56.6% | 0.0% (words) | 39.5% | 16.79 |
| Dog topic, coarse ×32 | 14.5% | 1.6% | 43.0% | 57.0% | 7.0% | 40.7% | 16.33 |
| Dog topic, middle ×32 | 14.5% | 2.6% | 42.3% | 57.3% | 10.9% | 39.6% | 16.35 |
| Dog topic, fine ×32 | 15.0% | 5.1% | 42.7% | 57.0% | 15.4% | 39.5% | 16.52 |
| **Dog topic, all ×32** | 14.4% | **0.0%** | 43.0% | 57.3% | **0.5%** | 40.9% | **16.33** |

Findings: (1) **The effects are specific, not disruption.** Random locks of identical size and usage leave every measure within noise (largest change: she/her +4.6 points on the random top lock, borderline). (2) **Each targeted lock only hits its own category:** Lily's locks cut girls but not dogs; Spot's and the dog-topic locks cut dogs but not girls. (3) **Dog stories behave like the beach topic:** coarse or all levels remove them; the all-level lock (384 codes) removes dog stories (17.4% → 0.5%) and Spot with no change to girls, boys or fluency. (4) **Spot's locks look like the dog-topic locks, Lily's do not look like a girl-topic lock**: Lily's middle lock separates the name from the girl, Spot's middle lock does not separate the name from the dog. Why is open; measuring code overlap between each name and its character type is the next test.


### J. Step 3: dog names share codes on the middle levels; girl names share theirs on the top levels
For 13 names (6 girls, 4 boys, 3 dogs; "Rex" skipped, 24 stories), the codes used at that name's positions were collected per level (1,200 positions per name, subsampled for fairness) and compared by histogram overlap (0 = no codes in common, 1 = identical use). Each name compared with itself in two halves gives the ceiling ("self": 0.44–0.50 on top levels, 0.61–0.68 middle/bottom). "Kind signal" = overlap with same-kind names minus overlap with other kinds.

| Level group | Girls' kind signal | Boys' kind signal | Dogs' kind signal |
|---|---|---|---|
| Top L0–3 | +0.03 to +0.08 (Lily +0.06) | +0.00 to +0.03 | +0.01 to +0.05 (Spot +0.03) |
| **Middle L4–7** | +0.00 to +0.04 (Lily +0.03) | −0.02 to +0.02 | **+0.08 to +0.11 (Spot +0.10)** |
| Bottom L8–11 | +0.01 to +0.04 | +0.00 to +0.02 | +0.02 to +0.04 |

Max is a dog in ~80% of its stories; on the middle levels it overlaps with dog names (0.30) more than with boys (0.22) or girls (0.21). Bottom level L11 overlap is ~0 for every name (one code per token, as expected).

**This matches the lock results.** On the middle levels dog names draw on shared "dog" codes, so locking Spot there also weakens dogs; girl names draw mostly on their own codes there, so Lily's middle lock removes Lily but keeps girls. On the top levels girl names share codes, matching Lily's top lock removing girls. It does not explain why Spot's *top* lock removes dogs (dogs' top-level kind signal is small). Effects are small (≤0.11 against a ~0.65 ceiling), there are only 3 dog names, and no error bars yet.


### K. Step 4: predictions for two names never locked before (written before running)
Middle-level locks (L4–7, 256 codes, same recipe). Prediction from section J: Max (shares codes with dog names) → Max and dog stories drop; Sue (mostly its own codes) → Sue drops, girl stories stay.

| Lock | Max | Sue | Dog stories | Girl hero (49-name list) | "a (little) girl" | Boys | she/her | genPPL |
|---|---|---|---|---|---|---|---|---|
| Normal | 6.7% | 11.6% | 17.4% | 47.9% | 31.1% | 51.9% | 39.5% | 16.39 |
| **Max middle** | **2.7%** | 11.9% | **13.3%** | 47.2% | 31.9% | 53.4% | 39.5% | 16.53 |
| **Sue middle** | 7.0% | **3.0%** | 18.1% | **43.4%** | **26.9%** | 50.7% | 39.0% | 16.44 |
| (ref) Lily middle | 6.7% | 13.0% | 17.7% | 45.4% | 28.5% | 51.5% | 40.2% | 16.56 |

**Max: prediction held** (Max −4.0 points, dog stories −4.1, boys/girls unchanged). **Sue: half held.** Sue dropped from 11.6% to 3.0% with nothing else touched, but girl-hero stories also fell by ~4.5 points, even with a 49-name list (Sue lost 64 "named" stories; other girls' names gained ~36: Sally, Lily, Lucy, Amy, Jane…). So roughly 6 in 10 Sue stories kept a named girl. Lily's middle lock shows a similar but smaller dip. The overlap rule gets the direction right for both names but under-predicts how much girl-ness the girl names carry. The first 14-name girl list also under-counted girls (42.3% vs 47.9% with 49 names); earlier "girl hero" numbers are comparable to each other but are lower bounds.


### L. Scale: the same lock in three generator sizes on real news (LM1B)
LM1B small / medium / large generators (12 / 24 / 36 layers; 97M / 317M / 725M) share one tokenizer (12 levels × 8,192 codes, BERT-uncased text, 128-token chunks). Our P40 replication of their quality: small 93.15, medium 88.36, large 81.12 genPPL. Target "obama" (token 8112; 7.3% of real validation chunks). Audit on 1,500 real chunks vs 4,000 background (2 rounds × 32 codes/level, the Lily recipe; tokenizer-side removal 57% after round 2, again showing self-healing). The **same** 748-code lock, and a usage-matched random lock of the same size, were applied to all three generators.

| Generator | Arm | obama | president | other politicians | government | police | game/team | genPPL |
|---|---|---|---|---|---|---|---|---|
| Small | normal | 11.6% | 8.4% | 8.8% | 9.1% | 5.4% | 10.4% | 93.15 |
| Small | **lock, all** | **0.2%** | 0.1% | 0.9% | 8.6% | 6.2% | 15.0% | 99.94 |
| Small | random, all | 13.4% | 9.2% | 10.5% | 7.2% | 3.7% | 6.9% | 103.90 |
| Small | lock, mid+fine | 0.2% | 0.3% | 1.6% | 7.6% | 5.0% | 11.9% | 100.19 |
| Medium | normal | 15.9% | 11.6% | 15.4% | 12.0% | 9.8% | 11.2% | 88.36 |
| Medium | **lock, all** | **0.2%** | 0.8% | 2.7% | 11.2% | 10.2% | 12.8% | 95.64 |
| Medium | random, all | 16.2% | 14.1% | 16.5% | 11.8% | 7.1% | 8.4% | 96.86 |
| Large | normal | 14.5% | 10.0% | 15.1% | 13.4% | 10.4% | 12.5% | 81.12 |
| Large | **lock, all** | **0.2%** | 0.6% | 2.2% | 10.2% | 10.7% | 12.9% | 87.38 |
| Large | random, all | 16.0% | 11.8% | 14.8% | 11.4% | 8.8% | 9.4% | 90.10 |
| Large | lock, mid+fine | 0.5% | 1.1% | 3.8% | 11.9% | 11.1% | 12.0% | 87.22 |

**Pipeline control** (small generator): unpatched rerun 93.15; patch installed with an empty lock 93.15 (identical on every seed); random lock on L4–11 only 102.77. So the quality cost is real, not a code artifact.

Findings: (1) **The lock holds across a 7.5× range of generator size**: 0.2% at every size; no sign that larger generators route around it. (2) On news, "obama" behaves like Spot, not Lily: the lock also removed "president" and other politicians (a "politicians" topic lock), while government, police and sports stayed. (3) **On real news, locking ~500–750 codes costs ~7–10% genPPL even when the codes are random** (and random costs slightly more than the targeted lock); on TinyStories the same-size random locks cost nothing. Whether a smaller or more selective lock avoids this cost is open.

## Findings
1. **Self-healing.** In NCM's residual hierarchy, removing a detail from some levels is undone by the others, so tokenizer-side masking needs every level and plateaus near 80%.
2. **A plan-level lock beats a word ban.** Because the generator plans coarse-to-fine, forbidding codes lets it plan around the missing name instead of leaving holes. A word ban cannot do this.
3. **Where to lock matters, and it is not where we guessed.** We predicted the name was a fine detail. It is a **middle-level decision**: coarse levels carry the story type (locking them removes the girl, not the name), middle levels carry the name, fine levels carry it to the page.
4. **Fluency metrics miss the real cost.** The all-level lock scored as well as normal on genPPL while erasing a third of the girl-hero stories. Content checks are required, not just perplexity.
5. **There is a floor.** About 500 codes (1.0%) gets Lily to ~1.2%; more codes cost content without lowering the floor. The tail leaks back mid-story.
6. **The effects are specific.** Usage-matched random locks of the same size change nothing on TinyStories; each targeted lock only affects its own category.
7. **Whether a detail can be locked alone depends on code sharing.** Names that share middle-level codes with their kind (dog names) take the kind with them; names that mostly own their codes (girl names) can be removed while the kind stays. This predicted Max correctly and Sue in direction.
8. **A topic lock removes the topic; a word ban hides the words** (LLM judge: 0.0% vs 5.2% hidden beach stories).
9. **The lock scales across generator sizes** (97M → 725M), but on real news it costs ~7–10% fluency.
10. **Remembering is harder than forgetting.** A lock removes a choice; teaching a new name needs coordinated new choices at many levels.

## Limitations
Mostly one dataset (TinyStories) plus one real-text test (LM1B, one target); generators of 88M–725M parameters, far below chat-model scale; 1,280 samples per arm (margins of error ≈ ±2–3 points at 15–40%, ±1 point near 5%); few names per kind (3 dog names). "Girl hero" is a name-list heuristic. genPPL uses a single judge (GPT-2 large). The lock is a mask at inference, not deletion: the weights still contain the information, and anyone holding the weights can remove the mask. Adversarial prompting and relearning attacks were not tested (generation here is unconditional). The LLM judge (Qwen3:8b) caught 73% of word-list beach stories. Name-list counts of girl heroes are lower bounds.

## Prior art this builds on
- Discrete Key-Value Bottleneck for continual learning (Träuble et al., ICML 2023, arXiv:2207.11240), adapted to NLP (Diera et al., arXiv:2412.08528).
- Unlearning by masking discrete codes (Shah et al., TMLR 2025, arXiv:2311.15268) and CodeUnlearn (Wu et al., arXiv:2410.10866).
- Sparse memory finetuning with TF-IDF slot selection (arXiv:2510.15103).
- **Closest prior work, from coarse-to-fine image models (VAR, NCM's image cousin):** ScaleErasure (ICML 2026, arXiv:2606.29282) erases concepts at inference time by steering logits only at **intermediate scales**, excluding coarse scales because they shift global structure and fine scales because they add little. SACE (arXiv:2606.15819) finds concepts locked in at the coarsest scale and fine-tunes there. S-VARE/VARE (ICLR 2026, arXiv:2509.22400) fine-tune VAR for erasure. Our coarse/middle/fine pattern for a *name* matches ScaleErasure's image finding; the contribution here is showing it in **text**, with a different and cheaper mechanism (a hard code lock from an audit, no extra forward passes, an auditable mask file), a word-ban comparison that has no image analogue, and a name-vs-topic difference (names: middle levels; topics: coarse or all levels).
- Locating knowledge in the middle of a network has a well-known LLM parallel: ROME (Meng et al., NeurIPS 2022, arXiv:2202.05262) found factual associations mediated by middle-layer MLPs.
- Inference-time locks are known to be bypassable when an attacker can fine-tune the weights (relearning attacks, e.g. arXiv:2406.13356); this applies to our lock as well.
- What is new here, to our knowledge: the first forgetting study on a text coarse-to-fine model (NCM); self-healing of tokenizer-side masks across residual levels; a generation-time code lock vs. a word ban (the ban leaves holes and hides topics); and a level map in text (story type → coarse, name → middle, topic → coarse/all).

## Next
- A smaller or more selective lock on real news that avoids the ~7–10% fluency cost.
- Attack tests: plant a target's coarse-level plan and see if it returns; relearning by fine-tuning.
- More names and kinds, with error bars on the code-overlap measure; why Spot's coarse lock removes dogs.
- The remember half with replay (mixing old stories into training) to avoid over-use of the new name.

## Files
`scripts/` holds every run in order: `ncm1*.sh` (Phase 1 replication) · `ncm2r/2a/2b` (recon, audit, iterative masking) · `ncm3/3b/3c/3d` (generator lock, word ban, recount, per-level lock) · `ncm4` (middle-lock tuning) · `ncm5/5b` (beach topic, LLM judge) · `ncm6r/6` (teaching a new name) · `ncm7` (Spot) · `ncm8` (dog topic) · `ncm9` (placebo + first-level fix) · `ncm10` (code overlap) · `ncm11/11b` (predictions: Max, Sue) · `ncm12r/13/14/14c` (LM1B scale-up and pipeline control). `results/` holds each run's text report. Each script runs on the Dell from `/data/ncm/ncm` (the HazyResearch/ncm repo at c702742) with the released checkpoints.
