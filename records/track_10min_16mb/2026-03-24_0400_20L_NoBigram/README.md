# 20L NoBigram (Exp 56)

## 목적
극단적 깊이 실험. BigramHash 제거로 확보한 예산 + AttnOnly 레이어로 20L 도달.
깊이의 수확 체감 지점을 확인하기 위한 극단적 테스트.

## 구성
- Encoder layers 0-11: Attention-Only (MLP 없음, KV heads 2) — 12L
- Encoder layer 12: Mini (MLP 1.5x, KV heads 1) — 1L
- Decoder layers 13-19: Full (MLP 3.0x, KV heads 4) — 7L
- BigramHash: **제거**
- Pruning: 5%

## 예상 결과
- val_bpb: ~1.135~1.150
- 크기: ~16.0MB
- 위험: 매우 높음 (AttnOnly 12개 레이어가 충분한 표현력을 가지는지 불확실)

## 실제 결과
(학습 후 기록 예정)
