# Exp 159: TTT Chunk Size Tuning

## 목적
Chunk 크기를 줄이고 epoch를 늘려 mask 적응 빈도를 높임

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)

## 변경 사항
1. ttt_chunk_tokens: 32768 → 16384 (chunk 수 2배)
2. ttt_epochs: 3 → 5 (chunk당 더 많이 학습)
3. ttt_lr: 0.005 → 0.008 (작은 chunk의 noisy gradient 보상)
4. ttt_grad_clip: 1.0 → 0.5 (안정성)
5. LR schedule: warmup 3%, constant 64%, cosine 33%

## 예상 결과
- val_bpb: ~1.1165-1.1175 (더 빈번한 adaptation)
- eval 시간: ~590-650s (chunk 수 2배지만 각 chunk 학습은 동일)

## 실제 결과
_(실험 후 기록)_
