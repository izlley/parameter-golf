# Record candidate: 11L XSA + CaseOps + MP3 marker-pair fusion + SmearGate (alias-prev dampening) + LQER + swttt

**val_bpb: 1.06653** (1-seed measured on 8×H100 SXM, swttt eval) | ~15,979,621 bytes | 8×H100 SXM, 600 s wallclock | TTT (LoRA swttt)

5-seed verification on runpod (SEEDS=42, 0, 1234, 314, 999) — see `train_seed*.log` after the run.

## Headline change vs. 1.0810 leaderboard

CaseOps tokenizer (lossless lowercase + 4 reserved markers) + **MP3 marker-pair fusion**:
the three 2-grams `[▁, TITLE]` / `[▁, ALLCAPS]` / `[▁, CAPNEXT]` are fused into single
**alias donor tokens** (donors 8/9/10 from byte-fallback IDs that occur 0× in the
CaseOps corpus). The downstream **word X is preserved** — preventing the d=1
prediction collapse that broke earlier full-word-fusion attempts.

Token saving: **8.47 %** (8.05 B → 7.37 B train tokens; 47.85 M → 43.77 M val tokens).
Bytes lossless: val sidecar sum unchanged (151,080,645 bytes).

To prevent SmearGate from over-injecting (▁ + marker) information into the next
word, alias-prev positions get a **0.5× SmearGate dampening** (`ALIAS_PREV_SMEAR_SCALE=0.5`).

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4× (2048) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Attention | FA3, GQA 2:1 | Baseline |
| XSA | All 11 layers | [#478](https://github.com/openai/parameter-golf/pull/478) |
| RoPE | Partial (16/64 dims) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| LN scale | 1/√(layer+1) | [#315](https://github.com/openai/parameter-golf/pull/315) |
| QK Gain | per-head, init=5.25 | [#1276](https://github.com/openai/parameter-golf/pull/1276) |
| U-Net skips | encoder–decoder + skip gates | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Parallel decoder | 2-lane from layer 7+ | [#1530](https://github.com/openai/parameter-golf/pull/1530) |
| Depth recurrence | layers 3-5, 3× loop, frac ≥ 0.35 | [#1344](https://github.com/openai/parameter-golf/pull/1344) |
| Logit softcap | 30 | Gemma2 |
| **SmearGate** (BOS-fixed) | Position-mixing + **alias-prev 0.5× dampening** | [#1667](https://github.com/openai/parameter-golf/pull/1667) + this work (alias dampening) |
| **Polar-Express Newton-Schulz** | Muon, 5 steps | [#1344](https://github.com/openai/parameter-golf/pull/1344) → [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| **MIN_LR floor** | 0.10 | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| **LQER asymmetric int4** | rank=4, top_k=3, asym_group=64 | [#1797](https://github.com/openai/parameter-golf/pull/1797) |
| Tokenizer | sp8192 lossless caps caseops v1 reserved | [#1729](https://github.com/openai/parameter-golf/pull/1729) |
| **MP3 marker-pair fusion** | warm-init alias rows + tied embedding (no extra params) | **this work** |
| Quantization | GPTQ int6 (matrix) + int8 (embed) | [#535](https://github.com/openai/parameter-golf/pull/535) |
| Optimizer | Polar-Express Muon (matrix) + AdamW (embed, scalars) | [#399](https://github.com/openai/parameter-golf/pull/399) |
| Weight averaging | EMA(0.9965) + tight SWA (every 50) | [#401](https://github.com/openai/parameter-golf/pull/401) |
| Compression | brotli q=11 | [#160](https://github.com/openai/parameter-golf/pull/160) |
| **TTT (swttt)** | Sliding-window LoRA TTT — rank 32, layers 5-10, c_qkv module, lr=1e-4, B-norm cap=5.0, train_every=480 | [#461](https://github.com/openai/parameter-golf/pull/461) etc. |

## MP3 marker-pair fusion (vocabulary surgery)

Three patterns active in MP3, performed by `prepare_marker_pair_v3.py` on the
already-tokenized CaseOps shards:

```
[▁, TITLE]    -> ALIAS_SP_TITLE    (donor=8)
[▁, ALLCAPS]  -> ALIAS_SP_ALLCAPS  (donor=9)
[▁, CAPNEXT]  -> ALIAS_SP_CAPNEXT  (donor=10)
```

Donor IDs come from byte-fallback tokens that occur **0 times** in the CaseOps
corpus (verified by full-corpus token-count audit). They become alias rows after
warm-init + training; their original byte-fallback meaning is unused in CaseOps text.

### Static warm-init for alias rows

Each alias row is initialized as a weighted, norm-matched composite of its
constituent marker rows. This avoids cold-start within the 600 s training budget:

```
E[alias_donor] = 0.4·E[▁] + 0.6·E[marker], renormalized to ‖E[marker]‖
```

Hparams: `MARKER_PAIR_W_SPACE=0.4`, `MARKER_PAIR_W_TITLE=0.6`, `MARKER_PAIR_NORM_TARGET=title`.

### SmearGate alias-prev dampening

The PR #1667-style SmearGate mixes the previous-position embedding into the next
position. With marker-pair fusion, the alias position carries strong (▁ + marker)
information, and unconstrained smear over-injects this into the following word. We
dampen prev-alias smear by **0.5×** (`ALIAS_PREV_SMEAR_SCALE=0.5`).

Implemented as a per-position multiplicative mask applied inside the SmearGate path
(both `_forward_hidden` and `forward_ttt`); dampening only kicks in when
`alias_dampening_active` is set (Python-bool flag, set in `main()` to avoid
torch.compile data-dependent branching).

## Why this works (NLL distance breakdown vs. CaseOps base)

Per-position NLL bucketed by distance from the most recent alias position:

| distance | 1772 (MP1, single TITLE pair) | **1779 (MP3, 3-pair)** |
|---|---|---|
| d=1 (next word) | +0.159 mbpb | **−0.017** ✓ (preserved) |
| d=2             | +1.096        | +0.262 |
| d=3             | +0.301        | −0.003 |
| d=4–8           | −0.435        | **−1.199** ✓✓ |
| d=9–32          | −0.808        | **−1.687** ✓✓✓ |
| d=33+           | −0.100        | −0.247 |
| alias positions | −0.16         | +1.430 (TITLE alias dominates) |
| **GRAND TOTAL** (int6 level) | +0.05 | **−1.46** mbpb |

Insights:
- **d=1 is preserved** because word X is unchanged in the alias scheme — full-word-fusion
  variants paid heavy cost at d=1 (1.07 ‒ 2.78 mbpb) by absorbing the word into the alias.
- **d=4+ shows compounding wins** — shorter sequences let attention model long-range
  structure better.
- Alias positions themselves show a +1.43 mbpb cost (TITLE alias dominates); this is more
  than offset by the non-alias gains.

Per-marker breakdown (from int6 forward on val):
- TITLE alias (donor 8, 3.79 M occurrences): +1.349 mbpb
- ALLCAPS alias (donor 9, 297 K occurrences): +0.081 mbpb
- CAPNEXT alias (donor 10, ~0 occurrences): negligible

## Files

| File | Purpose |
|---|---|
| `train_gpt.py` | Full training script (~2.8 k lines). Configurable via env vars; defaults reproduce this submission. |
| `prepare_caseops_data.py` | CaseOps-tokenized FineWeb shard prep + per-token byte sidecar (~7 KB). |
| `prepare_marker_pair_v3.py` | Vocabulary surgery on top of CaseOps shards: fuses three `[▁, MARKER]` 2-grams into alias donor tokens. Writes `alias_map.json`. |
| `lossless_caps.py` | Bijective lowercase + private-use-area sentinel pre-encoding (CaseOps infrastructure, ~28 KB). |
| `tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model` | SentencePiece model used by CaseOps (~367 KB). |
| `alias_map.json` | MP3 alias map (donor ↔ marker mapping). |
| `requirements.txt` | Minimal Python deps. |
| `run_5seed.sh` | 5-seed runner. Outputs `train_seed{42,0,1234,314,999}.log`. |
| `train_seed*.log` | Per-seed run logs (created by `run_5seed.sh`). |

## Pipeline

### 0. System deps

```bash
apt-get install -y lrzip   # not required for this submission (uses brotli) but harmless
pip install -r requirements.txt
pip install --no-deps flash_attn_3 \
    --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
```

### 1a. Download raw FineWeb-10B docs (one-time, ~50 GB on disk, ~30 min)

The canonical `docs_selected.jsonl` is produced by the upstream parameter-golf
repo's data downloader (already included in this fork at `data/`):

```bash
# from the repo root (parent of records/)
python3 data/download_hf_docs_and_tokenize.py
# writes ./data/datasets/fineweb10B_raw/docs_selected.jsonl
```

### 1b. CaseOps tokenization (one-time, ~30-60 min)

Tokenizes `docs_selected.jsonl` into CaseOps shards plus the val byte sidecar.

```bash
# from this submission directory
python3 prepare_caseops_data.py \
    --docs ../../../data/datasets/fineweb10B_raw/docs_selected.jsonl
# writes ./data/datasets/fineweb10B_sp8192_lossless_caps_caseops_v1_reserved/
#   fineweb_train_*.bin
#   fineweb_val_*.bin
#   fineweb_val_bytes_*.bin
```

See the docstring in `prepare_caseops_data.py` for the full CLI.

### 2. Vocabulary surgery (MP3)

Reads the CaseOps shards from step 1, fuses the three 2-gram patterns into alias
donor tokens, and writes a new dataset directory with byte sidecar updated.

```bash
python3 prepare_marker_pair_v3.py
# writes ./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3/
#   fineweb_train_*.bin
#   fineweb_val_*.bin
#   fineweb_val_bytes_*.bin
#   alias_map.json
```

### 3. 5-seed training (~5 × 10 min wallclock + per-seed eval)

```bash
bash run_5seed.sh
# writes train_seed{42,0,1234,314,999}.log in this dir
```

Each seed does its own 600 s training + swttt evaluation in a single process.
Final summary table is printed at the end.

### Single-seed reference command

```bash
SEED=42 \
CASEOPS_ENABLED=1 \
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops_marker_pair_v3 \
TOKENIZER_PATH=./tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model \
MARKER_PAIR_MODE=1 \
MARKER_PAIR_W_SPACE=0.4 MARKER_PAIR_W_TITLE=0.6 MARKER_PAIR_NORM_TARGET=title \
ALIAS_PREV_SMEAR_SCALE=0.5 ALIAS_CURR_SMEAR_SCALE=1.0 \
TTT_ENABLED=1 TTT_LR=0.0001 TTT_MAX_B_NORM=5.0 \
MIN_LR=0.10 \
LQER_ENABLED=1 LQER_RANK=4 LQER_TOP_K=3 LQER_ASYM_ENABLED=1 LQER_ASYM_GROUP=64 \
SMEAR_GATE=1 SMEAR_GATE_WIDTH=12 SMEAR_TARGET=fast SMEAR_LANE_INIT=clean SMEAR_LR_MULT=2.0 \
ATTN_OUT_GATE=0 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Hardware / environment

- 8 × H100 80 GB SXM (NVLink), 600 s wallclock budget
- PyTorch 2.9.1+cu128, CUDA 12.8, FlashAttention 3 (Hopper kernels)

## Notes

- This submission uses **brotli q=11** for compression (no system-level lrzip
  needed), unlike PR #1797's per-group lrzip pipeline. Compression is monolithic
  on the int6 GPTQ blob plus int8 embedding plus state-dict scaffolding.
- The 600 s budget includes GPTQ calibration; `GPTQ_RESERVE_SECONDS` is internally
  managed by `train_gpt.py`.
- TTT here is `swttt` (sliding-window TTT), not phased TTT — LoRA adapters are
  trained on a sliding window over the val docs and applied during the same
  forward pass at eval time. See `train_gpt.py` for details.

## Submission status

| Seed | val_bpb (swttt) | val_loss | size (bytes) | step_avg | steps |
|---|---|---|---|---|---|
| 42 | TBD (5-seed run on runpod) | TBD | TBD | TBD | TBD |
| 0 | TBD | TBD | TBD | TBD | TBD |
| 1234 | TBD | TBD | TBD | TBD | TBD |
| 314 | TBD | TBD | TBD | TBD | TBD |
| 999 | TBD | TBD | TBD | TBD | TBD |
| **Mean** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** |

(1-seed reference on author's DGX H100 with same code — seed 42:
val_bpb=1.06653, size=15,979,621 bytes, steps=4455, step_avg=134.69 ms,
note runpod SXM tends to give slightly faster steps and tighter `val_bpb`.)
