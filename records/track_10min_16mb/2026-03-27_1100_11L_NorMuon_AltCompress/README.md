# Exp 134: NorMuon + Alternative Compressor (brotli)

## Purpose
NorMuon의 uniform weight 분포에서 zstd-22보다 brotli가 더 효율적인지 테스트.
zstd는 LZ77+FSE 기반, brotli는 LZ77+Huffman+context modeling으로 다른 특성.

## Base
Exp 121 (11L NorMuon, val_bpb 1.1183, artifact 17,158,779 bytes)

## Changes from Exp 121

### Brotli Compression
- `zstd-22` → `brotli (quality=11)` (최대 압축 레벨)
- Brotli는 static dictionary + context-dependent Huffman coding 사용
- Uniform 분포에서 zstd보다 나은 압축률 가능
- 압축 속도는 느리지만 직렬화는 1회만 수행하므로 무관
- Decompression은 빠름

### Fallback: LZMA
- Brotli 실패 시 LZMA (xz) 도 시도
- LZMA: range coding + Markov chain 기반, 높은 압축률

## Architecture (Exp 121과 동일)
- 11L, 512dim, 8H/4KV, GQA

## Expected Results
- val_bpb: **1.1183** (압축만 변경, 학습 동일)
- artifact: 미정 (brotli vs zstd 비교 목적)
- 크기 제한: 미확정
