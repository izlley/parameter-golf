# Exp 117: 11L Entropy-Guided Progressive Attention Looping (SGT)

## Purpose
Exp 93 아키텍처에서 SGT(Sparse Growing Transformer) 논문의 핵심 설계를 구현:
- **블록 전체 루핑 → high-entropy head만 선택적 루핑**으로 FLOPs 최소화
- **중간 레이어 고정 → deep-to-shallow 동적 선택**으로 수렴 안정성 확보
- **후반 20% 활성화 → 학습 중반부터 점진 성장**으로 수렴 곡선 자체를 낮춤

## Paper
[Sparse Growing Transformer](https://arxiv.org/abs/2603.23998)

## Base
Exp 93 (11L GatedAttn PerHead, val_bpb 1.1198 sliding_window)

## Changes from Exp 93

### 1. Entropy-Guided Attention Looping (논문 Eq.6~7)
- 블록 전체가 아닌 **각 레이어의 top-2 high-entropy head만** 선택적으로 루핑
- Attention head의 entropy: 마지막 토큰의 length-normalized Shannon entropy 사용
- 루핑 공식: `H(k) = H(k-1) + Σ_{i∈S} Attn(i)(H(k-1))`
- 추가 파라미터 없음 (동일 weight 재사용)

### 2. Progressive Growth Schedule (논문 Sec.4.2)
- **Warm-up Phase** (step < 500): 표준 Transformer로 학습, entropy 안정화 대기
- **Growing Phase** (step ≥ 500, Δt=250마다): entropy 기반 deep-to-shallow 레이어 선택
- **Fixed Phase**: 3개 레이어 도달 후 아키텍처 고정

### 3. SGT 목표 구성
| 항목 | 값 |
|---|---|
| 활성 루핑 레이어 수 (L) | 3 |
| 레이어당 루핑 head 수 (h) | 2 |
| warm-up 시작 step | 500 |
| 성장 간격 (Δt) | 250 |
| Entropy EMA alpha | 0.1 |

### 4. torch.compile 호환
- `torch._dynamo.config.recompile_limit = 32`
- 성장 시 `_sgt_active_layers` (frozenset) 변경으로 recompile 트리거 (최대 3회)
- `sgt_head_gate` 는 nn.Buffer → graph 구조 불변, 값만 변경

## Architecture
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.997) + U-Net skip
- **SGT Looping**: deep-to-shallow, top-2 high-entropy heads, L=3

## Exp 113 대비 변경 요약
| 항목 | Exp 113 (Block Loop) | Exp 117 (SGT) |
|---|---|---|
| 루핑 단위 | 블록 전체 | top-2 high-entropy head |
| 레이어 선택 | 고정 {2,3} | entropy 기반 동적 |
| 성장 방향 | 중간 레이어 | deep → shallow |
| 활성화 시점 | 후반 20%만 | step 500부터 점진적 |
| 속도 영향 | +18% (블록 2회) | attention-only 재실행 |

## Expected Results
- step_avg: ~95ms (초반) → ~97-100ms (3개 레이어 활성 후)
- Target: < 1.1198 bpb (Exp 93 대비 개선)
- Size: ~16.72MB (동일, weight sharing)
