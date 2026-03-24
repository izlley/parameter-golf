# Exp 88: Hymba-style Hybrid Head (Attention + SSM)

## Purpose
Hymba 논문에서 제안한 hybrid head 구조 실험. 동일 layer 내에서 일부 head는 attention으로, 나머지 head는 SSM(State Space Model)으로 처리하여 두 메커니즘의 장점을 결합.

## Base
Exp 73 (11L, 512dim, 8 heads, 4 KV heads GQA, int6+zstd)

## Architecture

```
Input x (B, T, 512)
  |
  +---> Attention Path (4 heads, 2 KV heads GQA)
  |       Q: (512 -> 256), K: (512 -> 128), V: (512 -> 128)
  |       RoPE + QK-norm + Flash Attention + XSA
  |       Output: (B, T, 256)
  |
  +---> SSM Path (4 heads worth = 256 dim)
  |       Proj_in: (512 -> 256)
  |       Conv1d(depthwise, k=4) + SiLU
  |       Parallel scan SSM (d_state=16)
  |       Gated output
  |       Output: (B, T, 256)
  |
  +---> Concatenate -> (B, T, 512)
  |
  +---> Output projection (512 -> 512)
```

## Changes

| Parameter | Value | Description |
|-----------|-------|-------------|
| HYMBA_ENABLED | 1 | Hymba hybrid head 활성화 |
| HYMBA_SSM_HEADS | 4 | SSM으로 처리할 head 수 (8개 중 4개) |
| HYMBA_D_STATE | 16 | SSM state dimension |
| HYMBA_D_CONV | 4 | Conv1d kernel size for SSM |

## Approach
- 8개 head 중 4개는 기존 GQA attention (RoPE, QK-norm, Flash Attention)
- 나머지 4개는 SSM으로 대체 (Conv1d -> parallel scan SSM with gating)
- SSM은 `torch.compile(fullgraph=True)` 호환을 위해 Python for-loop 대신 log-space cumsum 기반 parallel scan 사용
- Attention과 SSM 출력을 concatenate 후 단일 output projection
- VE(Value Embedding)는 attention path에만 적용 (SSM path는 K,V가 없으므로)

## Parameter Savings
- Attention heads 4->4: Q proj (512->256), K proj (512->128), V proj (512->128) = 256K params/layer
- SSM heads: Proj_in (512->256) + SSM internal params ~= 140K params/layer
- 기존 대비 약간의 파라미터 절감 예상 (SSM이 KV projection보다 compact)

## Risk: HIGH
- `torch.compile(fullgraph=True)` 호환성 -- parallel scan의 log-space cumsum이 graph break 없이 동작하는지
- SSM의 numerical stability (log-space cumsum에서 exp overflow 가능)
- step_avg 증가 (SSM 연산 추가)
- 파라미터 수 변화로 인한 quantization budget 영향

## 환경변수
```bash
HYMBA_ENABLED=1       # 1=활성화, 0=비활성화 (fallback to pure attention)
HYMBA_SSM_HEADS=4     # SSM head 수
HYMBA_D_STATE=16      # SSM state dimension
HYMBA_D_CONV=4        # Conv1d kernel size
```

## 실제 결과
(학습 후 기록 예정)
