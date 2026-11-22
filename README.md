# eighty-six

**Taking names and topics "off the menu" of a Next-Concept Model, on a 2016 GPU.**

*To "86" something, in kitchen slang, is to take it off the menu.*

[Next-Concept Modeling (NCM)](https://arxiv.org/abs/2610.14162) (Hazy Research / Together AI, Oct 2026) writes text
coarse-to-fine: a tokenizer turns text into 12 levels of discrete codes, from a rough sketch down to every word, and a
generator writes one level at a time. This project asks: **can you make an NCM forget something by forbidding the codes
it uses, and where in the hierarchy do different kinds of things live?**

Everything here ran on one **Tesla P40 (2016, 24 GB, plain FP32)** using Stanford's released checkpoints.

## Results at a glance

| Experiment | Result |
|---|---|
| Replication (Phase 1) | Released checkpoints reproduce the paper on the P40: Sudoku 99.4 / 96.4 / 86.3% solved (paper 99.5 / 96.5 / 86.1), LM1B-small genPPL 93.15 (paper 93.05). TinyStories ~6% worse (open). |
| Lock a name (TinyStories) | Forbidding ~500 audited codes on the middle + fine levels: stories with "Lily" **14.9% → 1.3%**, girl heroes mostly kept under other names. |
| Lock vs. word ban | The word ban leaves holes ("a little girl named.. She…") and costs more fluency; the code lock plans around the missing name. |
| Lock a topic | Beach stories (LLM judge) **12.0% → 0.0%**; dog stories **17.4% → 0.5%**. A word ban only hid the words: 5.2% of stories were still beach stories ("pretty seaweed", "the deep seas, the reef"). |
| Placebo | Usage-matched random locks of the same size changed nothing on TinyStories. |
| Where things live | Story type → coarse levels. A name → middle levels, *unless* it shares codes with its kind (dog names do; girl names mostly don't), in which case the kind goes too. Predictions for two new names (Max, Sue) held in direction. |
| Scale (real news, LM1B) | The same lock removed "obama" in **97M, 317M and 725M** generators (≈15% → 0.2%); larger generators did not route around it. It also removed "president" and other politicians, and on real news any lock of this size (random ones too) cost ~7–10% genPPL. |
| Remembering | Teaching a new name by training only per-level output heads failed; a full fine-tune learned it but over-used it. Remembering is harder than forgetting. |

Full write-up with every table, sample and caveat: **[REPORT.md](REPORT.md)**. Phase 1 replication: [PHASE1_REPORT.md](PHASE1_REPORT.md).

## What is new, and what is not

Coarse-to-fine *image* models (VAR) already have concept-erasure work, including inference-time logit steering at
intermediate scales ([ScaleErasure](https://arxiv.org/abs/2606.29282), [SACE](https://arxiv.org/abs/2606.15819),
[S-VARE](https://arxiv.org/abs/2509.22400)); our name result matches their "intermediate scales" finding. As far as we
found, what is new here is: the first forgetting study on a **text** coarse-to-fine model; a hard code lock found by an
audit (no extra forward passes, an inspectable mask file); the word-ban comparison; self-healing of tokenizer-side masks;
the name-vs-topic and code-sharing results; a placebo control; and the scaling test across generator sizes.

## Limits (please read)

- Small models (88M–725M), far below chat-model scale; mostly one dataset (TinyStories) plus one real-text target.
- The lock is applied at inference. The knowledge is still in the weights; anyone with the weights can remove the lock.
- No adversarial or relearning attacks were tested. Generation is unconditional.
- 1,280 samples per arm: margins of error are about ±2–3 points at 15–40% and ±1 point near 5%.
- Content is measured with word/name lists (lower bounds) and one local LLM judge (Qwen3:8b, 73% recall vs. word lists).

## Reproducing

1. Clone [HazyResearch/ncm](https://github.com/HazyResearch/ncm) (we used commit `c702742`) to `/data/ncm/ncm` and
   download the checkpoints and datasets from Hugging Face (`hazyresearch/ncm-checkpoints`, `hazyresearch/ncm-datasets`).
   `scripts/ncm1.sh` does this.
2. Run the scripts in `scripts/` in order (`ncm1` → `ncm14c`). Each one writes a text report to `/data/lab/<name>/`.
   Longer runs have a `*go.sh` launcher that runs them in the background.
3. Our reports for every run are in `results/<name>/REPORT.txt`.

Hardware: Dell R740, Tesla P40, PyTorch 2.14 + CUDA 12.6, FP32, no `torch.compile`. Runs patch
`scripts/eval_shard.py` with a guarded hook (inactive unless `NCM_MASK` or `NCM_BAN` is set) and restore the original
file afterwards.

## Credits

Built on [Next-Concept Modeling](https://arxiv.org/abs/2610.14162) by Hermann Kumbong, Simran Arora, Azalia Mirhoseini,
Christopher Ré and co-authors, and their MIT-licensed code and checkpoints. Some recon reports in `results/` contain
excerpts of that code. Research and writing assisted by Claude (Anthropic).

**Author:** Khai Bustos ([KhaiB10](https://github.com/KhaiB10)). License: MIT.
