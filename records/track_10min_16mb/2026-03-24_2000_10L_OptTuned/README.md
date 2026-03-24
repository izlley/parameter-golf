# 10L Optimizer Tuned (Exp 72)

## 목적
Weighted SWA (Exp 65, val_bpb 1.14136)를 베이스로 optimizer 하이퍼파라미터 3가지를 동시에 조정.
양자화 내성 강화 + 학습 안정성 향상 + SWA 커버리지 확대.

## 변경 사항 (vs Exp 65 Weighted SWA)

| 파라미터 | Exp 65 (SOTA) | Exp 72 | 근거 |
|----------|---------------|--------|------|
| weight_decay | 0.04 | **0.05** | 가중치 magnitude 억제 → 양자화 에러 감소 |
| β2 (AdamW) | 0.95 | **0.98** | 더 긴 second moment 평활화 → 학습 안정성 |
| swa_start_frac | 0.50 | **0.40** | 더 빠른 SWA 시작 → 더 많은 checkpoint 수집 |

## 추가 수정
- Muon weight_decay도 하드코딩 0.04 → `args.weight_decay` 참조로 통일
- wandb run name을 실험 디렉토리명 기반으로 변경

## 설계 근거
- **WD=0.05**: int5/int6 양자화 시 큰 가중치가 클리핑되면 에러 발생. WD를 높이면 가중치 범위가 좁아져 양자화 품질 향상. SOTA 어블레이션에서 WD=0.04가 0.03보다 나았으므로 0.05 시도.
- **β2=0.98**: 기본값 0.999 대비 0.95는 빠른 적응에 유리하지만, 짧은 학습(7100 steps)에서 gradient noise가 크면 더 안정적인 0.98이 도움. LLM 학습에서 0.95~0.99 범위가 일반적.
- **SWA 40%**: 기존 50%에서 10% 앞당김. ~7100 steps 기준 step 2840부터 수집 시작 → ~85개 checkpoint (vs 기존 35개). Weighted SWA에서 더 많은 checkpoint = 더 부드러운 평균.

## 예상 결과
- val_bpb: ~1.138~1.141 (각 개선이 0.001씩 기여 시)
- 크기: ~15.9MB (SOTA와 동일)
- 위험: 낮음 (세 변경 모두 보수적인 범위)

## 실제 결과
(학습 후 기록 예정)
