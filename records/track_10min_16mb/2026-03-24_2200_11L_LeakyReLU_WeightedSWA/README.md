# 11L LeakyReLU(0.5)² + Weighted SWA + EMA 블렌딩 (Exp 74)

## 목적
리더보드 SOTA #1 코드의 dead SWA 코드를 활성화하고 Weighted SWA로 개선.
LeakyReLU(0.5)²와 결합하여 이중 개선 달성.

## 베이스
`2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/train_gpt.py`

## 변경 사항

| 파라미터 | SOTA #1 | Exp 74 | 근거 |
|----------|---------|--------|------|
| MLP activation | relu² | **LeakyReLU(0.5)²** | dead neuron 제거 |
| SWA 수집 | uniform (dead code) | **Weighted (linearly increasing)** | 최신 checkpoint에 높은 가중치 |
| SWA 적용 | 미적용 (EMA만 사용) | **EMA + Weighted SWA 블렌딩 (0.5:0.5)** | 두 averaging 기법 상보적 활용 |

## 설계 근거
- **SOTA #1의 SWA는 dead code**: SWA 상태를 수집하지만 최종 모델에 적용하지 않음. EMA만 적용됨.
- **Weighted SWA**: Exp 65에서 uniform SWA 대비 -0.001 bpb 검증. 선형 증가 가중치로 최신 checkpoint 강조.
- **EMA + SWA 블렌딩**: EMA는 전체 학습에 걸친 smooth averaging, SWA는 warmdown 후반부 집중. 0.5:0.5 블렌딩으로 두 관점 결합.

## 예상 결과
- val_bpb: ~1.1190~1.1220
- 크기: ~15.55MB (변경 없음)
- 위험: 매우 낮음

## 실제 결과
(학습 후 기록 예정)
