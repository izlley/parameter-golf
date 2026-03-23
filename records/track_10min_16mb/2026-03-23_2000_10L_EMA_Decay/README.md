# 10L EMA Decay (Exp 48)

## 목적
SOTA(10L, val_bpb 1.14239)의 SWA(균등 평균)를 EMA(지수 이동 평균)로 교체.
SWA는 모든 체크포인트에 동일 가중치를 주지만, EMA는 최근 가중치에 더 큰 가중치를 부여하여 학습 후반의 더 좋은 가중치를 반영.

## 변경 사항 (vs SOTA 10L)
- SWA → **EMA** (decay=0.999)
- ema_start_frac: 0.50 (warmdown의 50% 지점부터 EMA 시작)
- 매 스텝 업데이트: `ema = decay * ema + (1-decay) * current`

## EMA vs SWA 비교
- **SWA**: N개 체크포인트 균등 평균 (1/N 가중치)
- **EMA**: 최근 체크포인트에 지수적으로 높은 가중치 (decay=0.999 → 최근 1000스텝 영향)
- Exp 40 (LogitEnsemble)에서 SWA + last의 앙상블이 효과적 → SWA가 과도하게 평활화할 가능성 시사
- EMA는 최근 가중치 비중이 높아 이 문제를 완화할 수 있음

## 예상 결과
- val_bpb: ~1.140~1.143
- 크기: ~15.8MB (SOTA와 동일)
- 위험: 낮음 (SWA와 같은 역할, 다른 가중치 스킴)

## 실제 결과
(학습 후 기록 예정)
