# Exp 133: NorMuon + BigramHash 4096

## Purpose
BigramHash vocab을 8192 → 4096으로 줄여 embedding 크기 절감.
BigramHash embedding은 int8 양자화되므로 vocab 절반 = 직접적 크기 절감.

## Base
Exp 121 (11L NorMuon, val_bpb 1.1183, artifact 17,158,779 bytes)

## Changes from Exp 121

### BigramHash 4096
- `bigram_vocab_size`: 8192 → 4096
- Embedding 파라미터: 8192×128 → 4096×128 (524,288 → 262,144 elements, -262K)
- int8 양자화 + scale 기준으로 ~262KB raw 절감
- zstd 압축 후 ~100-150KB 절감 예상
- 해시 충돌 증가로 bpb 약간 악화 예상

## Architecture
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- **BigramHash 4096** (변경), VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15
- NorMuon

## Expected Results
- val_bpb: ~1.1195-1.1210 (+0.001~0.003)
- artifact: ~16.7-17.0MB (단독으로는 부족할 수 있음)
- 크기 제한: 단독 통과 어려움, 다른 기법과 조합 필요
