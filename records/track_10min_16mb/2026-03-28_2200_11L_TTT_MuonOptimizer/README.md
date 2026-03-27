# Exp 162: TTT Muon-Lite Optimizer

## 목적
LaCT 논문 아이디어 — SGD 대신 Muon 스타일 optimizer로 TTT mask 학습. Sign-based update로 방향만 사용하여 안정적 학습.

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)

## 변경 사항
1. SGD → MuonLiteTTT (sign-based + momentum + nesterov)
2. ttt_lr: 0.005 → 0.002 (sign update는 magnitude가 다르므로 LR 조정)
3. 1D 파라미터에 대한 Muon 근사: sign(g) * sqrt(N) / norm

## 예상 결과
- val_bpb: ~1.1165-1.1180 (더 안정적인 mask 학습)
- eval 시간: ~동일 (optimizer overhead 미미)

## 실제 결과
_(실험 후 기록)_
