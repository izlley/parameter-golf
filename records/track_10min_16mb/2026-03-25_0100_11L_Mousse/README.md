# 11L Mousse Optimizer + LeakyReLU(0.5)² (Exp 77)

## 목적
Muon optimizer에 per-parameter adaptive momentum을 추가한 Mousse로 학습 품질 개선.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

## 변경 사항

| 파라미터 | SOTA #1 | Exp 77 | 근거 |
|----------|---------|--------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거 |
| Optimizer | Muon (fixed momentum) | **Mousse (adaptive momentum)** | param별 최적 momentum |

## Mousse Optimizer 설계
- **베이스**: Muon (Newton-Schulz 5 iterations, Nesterov)
- **추가**: Gradient norm 분산 추적 (EMA, β=0.999)
- **적응 메커니즘**:
  - 각 파라미터의 gradient norm의 coefficient of variation (CV) 계산
  - CV가 높은 파라미터 (불안정) → momentum 감소 (더 반응적)
  - CV가 낮은 파라미터 (안정) → momentum 유지 (기존 Muon과 동일)
  - `effective_momentum = base_momentum - mousse_range * CV_clamped`
  - mousse_range=0.1: momentum 범위 0.89~0.99 (base=0.99 기준)
- **비용**: 파라미터당 2개 스칼라 (grad_norm EMA, grad_norm² EMA), GPU 메모리만

## 예상 결과
- val_bpb: ~1.1170~1.1220
- 크기: ~15.55MB (변경 없음)
- 위험: 중~높음 (학습 불안정 가능, step time 2-3ms 증가)

## 실제 결과
(학습 후 기록 예정)
