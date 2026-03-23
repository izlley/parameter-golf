# 18L NoBigram (Exp 55)

## 목적
BigramHash(10240, dim=128)를 제거하여 ~0.87MB 확보, 이를 추가 레이어에 투자.
16L(8AttnOnly+8Full)에서 18L(10AttnOnly+8Full)로 깊이 확장.
BigramHash의 -0.003~0.005 bpb 기여 vs 추가 2개 AttnOnly 레이어의 깊이 효과 비교.

## 구성
- Encoder 10L: Attention-Only (MLP 없음, KV heads 2)
- Decoder 8L: Full (MLP 3.0x, KV heads 4)
- BigramHash: **제거** (0.87MB 절약)
- Pruning: 7%

## 예상 결과
- val_bpb: ~1.135~1.145
- 크기: ~16.0MB
- 위험: 높음 (극단적 깊이 + bigram 제거의 상호작용 불확실)

## 실제 결과
(학습 후 기록 예정)
