# 11L Slim + Pruning 11% + Encoder MLP 2.75x (5L) + Bigram 9216

## 변경 사항 (vs Exp 34 Combined Slim)
- encoder_mlp_layers: 3 → 5
- Pruning: 10% → 11% (최소한의 pruning 증가)
- BigramHash: 10240 → 9216
- 예상 절약: ~400KB → 16.08MB

## 실제 결과
(학습 후 기록 예정)
