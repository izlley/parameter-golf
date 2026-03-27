# Exp 161: TTT Surprise-Weighted Training

## 목적
Titans 논문의 아이디어를 적용 — chunk 내에서 높은 loss(놀라운) 시퀀스에 더 큰 gradient 가중치를 부여하여 mask 학습 효율 향상

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)

## 변경 사항
1. Phase 1 후 현재 chunk의 per-sequence surprise(NLL) 계산
2. Softmax(surprise/0.5)로 가중치 생성 (temperature=0.5)
3. Phase 2에서 weighted loss로 mask 학습
4. 높은 loss 시퀀스 → 더 큰 gradient → mask가 어려운 패턴에 집중

## 예상 결과
- val_bpb: ~1.1165-1.1175 (어려운 패턴에 집중 학습)
- eval 시간: +10-15% (surprise 계산 오버헤드)

## 실제 결과
_(실험 후 기록)_
