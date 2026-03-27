# Exp 129: 11L NorMuon + 6-bit BitPack

## Purpose
Exp 121 NorMuon의 best bpb (1.1183)를 유지하면서, 6-bit 비트패킹으로 아티팩트 크기 초과 문제를 해결.

## Base
Exp 121 (11L NorMuon, val_bpb 1.1183 sliding_window, artifact 17,158,779 bytes — 381KB 초과)

## Changes from Exp 121

### 1. 6-bit BitPacking
- **기존**: int6 양자화 값(-31~31)을 int8(8비트)에 저장 → zstd-22 압축
- **변경**: int6 값 4개를 3바이트에 패킹 (6×4=24비트=3바이트) → zstd-22 압축
- `pack_int6`: signed [-31,31] → unsigned [0,62] → 4개씩 3바이트로 패킹
- `unpack_int6`: 역변환, deserialization 시 사용
- Raw 크기 25% 절감 (int8 대비), zstd 압축 후 추가 절감 기대

### 2. Serialization 포맷 변경
- `mixed_quantize_int6`: int6 텐서를 `.q` (int8) 대신 `.packed` (uint8 bitpacked)로 저장
- `dequantize_mixed_int6`: `.packed` + meta(n, shape)에서 복원
- int8 양자화 텐서(embed 등)는 기존 방식 유지
- `torch.save` / `torch.load` 호환 유지

## Architecture (Exp 121과 동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997), LN Scale, Late QAT 0.15
- **NorMuon**: neuron-wise L2 normalization after NS5

## Expected Results
- val_bpb: **1.1183** (NorMuon과 동일 — 학습/양자화 동일, 직렬화만 변경)
- artifact: **~14-15MB** (17.2MB 대비 ~2MB 절감 예상)
- step_avg: ~95ms (동일)
- 크기 제한: **통과 예상**
