# Exp 156: TTT Overlapping Chunks

## 목적
Phase 2 학습 시 이전 chunk의 후반 50%를 overlap하여 chunk 경계의 context 단절 문제 해결

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)

## 변경 사항
1. Phase 2 학습 범위: 현재 chunk만 → 이전 chunk 후반 50% + 현재 chunk
2. ttt_epochs: 3 → 2 (학습 데이터 1.5배 증가 보상)
3. Scoring (Phase 1)은 변경 없음 — legal 유지

## 예상 결과
- val_bpb: ~1.1165-1.1175 (mask 안정화로 -0.001~0.002 개선)
- eval 시간: ~490s (epoch 감소로 총 시간 유지)

## 실제 결과
_(실험 후 기록)_
