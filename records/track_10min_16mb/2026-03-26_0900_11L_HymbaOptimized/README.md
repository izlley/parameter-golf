# Exp 108: 11L Optimized Hymba (Selective Layers, Direct Concat)

## Purpose
Exp 88 (Hymba)은 per-step 성능이 가장 좋았지만 step_avg ~150ms로 4000스텝밖에 학습 불가. 3가지 핵심 최적화로 step_avg를 ~100ms로 줄여 6000+스텝 학습을 목표.

## Base
Exp 88 (11L Hymba Hybrid Head, val_bpb 1.20171 @step2500 — best per-step)

## Optimizations from Exp 88

### 1. SSM Heads 4→2 (SSM 연산 50% 감소)
- 8 heads 중 2개만 SSM, 6개는 attention
- SSM 처리 차원: 256→128

### 2. Direct Concatenation (프로젝션 2개 제거)
- 기존: attn(256)→proj_up(512) + ssm(256)→proj_out(512) → LN average → proj
- 변경: attn(384) + ssm(128) = 512 → proj (직접 concat)
- **제거**: `attn_proj_up`, `ssm_proj_out`, `attn_avg_norm`, `ssm_avg_norm`
- 레이어당 CastedLinear 2개 + LayerNorm 2개 삭제 → 레이어당 ~3ms 절감

### 3. Selective Layer Application (HYMBA_LAYERS)
- 기존: 모든 11개 레이어에 Hymba 적용
- 변경: 마지막 6개 레이어(5-10)만 Hymba, 0-4는 순수 attention
- 순수 attention 5레이어: 빠름 + Hymba 6레이어: 품질 유지

## Architecture
```
Layers 0-4: Pure CausalSelfAttention (GQA 8H/4KV) — fast
Layers 5-10: HybridAttention (6 attn + 2 SSM heads) — quality
```

## Expected Results
- step_avg: ~100-110ms (vs ~150ms Exp 88)
- Training steps: ~5500-6000 (vs ~4000 Exp 88)
- 더 많은 스텝으로 Exp 88의 per-step 품질 이점 활용
- Target: < 1.1220 bpb

## Environment Variables
```bash
HYMBA_ENABLED=1
HYMBA_SSM_HEADS=2          # 2 of 8 heads use SSM
HYMBA_D_STATE=16
HYMBA_D_CONV=4
HYMBA_LAYERS=5,6,7,8,9,10  # which layers use Hymba (empty=all)
```

## Dependencies
- `mamba_ssm` (Mamba1 CUDA selective scan)
- `LD_LIBRARY_PATH` must include CUDA 12.x runtime (auto-set in script)
