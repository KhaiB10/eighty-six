# Independent replication of Next-Concept Modeling (NCM), Phase 1: released checkpoints on a 2016 GPU

**Author:** Khai Bustos (GitHub: KhaiB10) · **Date:** 2026-11-11
**Paper:** "Next-Concept Modeling: Fast, Parallel Coarse-to-Fine Text Generation", arXiv:2610.14162 (Hazy Research / Together AI)
**Code:** github.com/HazyResearch/ncm @ c702742 (MIT) · **Weights:** huggingface hazyresearch/ncm-checkpoints

## Setup (what differs from the paper)
- **GPU:** one NVIDIA Tesla P40 (Pascal, 2016, 24 GB, no tensor cores, no native BF16).
- **Precision:** plain FP32 everywhere. The paper used BF16 mixed precision + torch.compile. No torch.compile here.
- **Software:** PyTorch 2.14.1 + CUDA 12.6, Ubuntu 24.04, NVIDIA driver 580.
- **Code:** the authors' evaluate.py and eval configs, unchanged. No code edits were needed to run in FP32.

## Results

| Task | Metric | Paper | Ours (P40, FP32) | Verdict |
|---|---|---|---|---|
| Sudoku easy (2,000 boards) | % fully solved | 99.5 | **99.40** (1988/2000) | replicates (within 1 std. error, ~0.2) |
| Sudoku medium (2,000) | % fully solved | 96.5 | **96.40** (1928/2000) | replicates (within 1 SE, ~0.4) |
| Sudoku hard (2,000) | % fully solved | 86.1 | **86.30** (1726/2000) | replicates (within 1 SE, ~0.8) |
| LM1B small (97M) | gen. perplexity (lower = better) | 93.05 | **93.15 ± 2.37** (5 seeds, 1,280 samples) | replicates |
| TinyStories | gen. perplexity | 15.41 | **16.39 ± 0.33** (5 seeds) | **~6% higher: open question** |
| TinyStories | entropy | 3.98 | 3.968 | matches (no repetition collapse) |

Sudoku cell accuracy: 99.97 / 99.79 / 98.35 % (easy / medium / hard). Every fully solved board was also a valid grid.

## Speed on the P40
- Sudoku: about 46 s per 2,000 boards (12 generation steps), so about 23 ms per board.
- TinyStories eval (5 seeds + GPT-2-large scoring): 187 s. LM1B small eval: 131 s.
- Peak GPU memory during the run: 22.5 GB. This includes GPT-2 large for scoring and another model resident on the GPU.

## Takeaways
1. **The released NCM checkpoints reproduce the paper's Sudoku and LM1B numbers on a 2016 GPU in plain FP32,** with no BF16 and no torch.compile.
2. **TinyStories is about 6% worse (about 3 seeds' standard deviation).** Candidate causes: FP32 vs BF16 numerics; a different GPT-2-large/transformers version used for scoring; the paper's sampling seeds. To be tested in Phase 1b.
3. **The repository ships evaluation code only, with no training code.** Phase 2 will write an open trainer and train the Sudoku model (25.5M generator, 14.9M tokenizer) from scratch on the same P40, following the recipe in the paper's Appendix C.
