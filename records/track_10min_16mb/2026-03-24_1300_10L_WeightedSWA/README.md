# 10L Weighted SWA (Exp 65)

## 목적
SOTA(10L, val_bpb 1.14239)의 SWA(균등 평균)를 가중 평균으로 교체.
SWA는 모든 체크포인트에 동일 가중치(1/N)를 부여하지만, Weighted SWA는 나중 체크포인트에 선형 증가 가중치를 부여.
Exp 48(EMA)은 지수적 가중치로 최근에 과도하게 편향 → 실패(+0.013). Weighted SWA는 uniform과 EMA의 중간.

## 변경 사항 (vs SOTA 10L)
- SWA 누적 시 가중치: uniform (1, 1, 1, ...) → **linear (1, 2, 3, ...)**
- `swa_weight_sum`으로 총 가중치 합 추적
- 최종 평균: `swa_state[name] / swa_weight_sum`
- 추가 파라미터 비용: **0** (코드 로직만 변경)

## 근거
- Exp 40 (LogitEnsemble): SWA + last 앙상블이 best val_bpb → SWA가 과도하게 평활화할 가능성
- Exp 48 (EMA 0.999): +0.013 bpb → 지수적 가중치는 너무 극단적
- Weighted SWA: 선형 증가로 최근 체크포인트 비중을 점진적으로 높임

## 예상 결과
- val_bpb: ~1.141~1.143 (uniform SWA와 유사하거나 약간 개선)
- 크기: ~15.9MB (SOTA와 동일)
- 위험: 매우 낮음 (코드 변경 최소)

## 실제 결과
(학습 후 기록 예정)
