# Exp 116: 11L HP Tuned + Size Fit (16MiB 이내)

## Purpose
Exp 109 (HP Tuned, sliding 1.1192)가 SOTA를 달성했으나 artifact size 16,994,165 bytes로 16MiB(16,777,216) 한도 초과. 파라미터 수를 줄이지 않고 양자화/압축 최적화로 size를 맞추면서 성능 유지.

## Base
Exp 109 (11L HP Tuned, sliding 1.1192, artifact 16.99MB — OVER LIMIT)

## Changes from Exp 109

### 1. Late QAT 조기 시작
| HP | Exp 109 | Exp 116 | 근거 |
|----|---------|---------|------|
| late_qat_threshold | 0.12 | 0.06 | QAT 2배 일찍 시작 → 양자화 친화적 weight 분포로 더 오래 학습 |

- Exp 109: lr_scale < 0.12 에서 QAT 시작 (step ~5952, 남은 ~356 steps)
- Exp 116: lr_scale < 0.06 에서 QAT 시작 (추정 step ~5700, 남은 ~600 steps)
- QAT 학습 시간 약 2배 → weight이 int6 양자화에 최적화

### 2. GPTQ clip percentile 확장
- **기존**: [0.9990, 0.9995, 0.9999, 0.99999, 1.0] (5개)
- **변경**: [0.998, 0.999, 0.9990, 0.9995, 0.9999, 0.99999, 1.0] (7개)
- 더 공격적인 outlier clipping 옵션 추가 → row별 최적 scale 탐색 범위 확대
- 양자화 오류 최소화하면서 int6 값 범위를 더 효율적으로 사용

## Architecture (동일)
- 11L, 512dim, 8H/4KV, GQA
- LeakyReLU(0.5)², MLP 3x, GatedAttn PerHead (bias=4.0)
- XSA last 4, Partial RoPE 16/64
- BigramHash 8192, VE layers 9,10
- EMA(0.998) + U-Net skip
- Muon lr=0.028, warmdown=3000, grad_clip=0.5

## Expected Results
- step_avg: ~95ms (동일)
- Target: ~1.1192 bpb (Exp 109와 동등 또는 약간 차이)
- Size: < 16,777,216 bytes (QAT 조기 시작 + 정밀 quantization으로 size 감소)
