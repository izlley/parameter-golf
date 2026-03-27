# Exp 146: NorMuon + EWQ (layers 0-2) + SupermaskTTT V3 (TTT B)

## 목적
실제 제출 가능한 최선의 조합: NorMuon(품질) + EWQ layers 0-2(크기 통과) + TTT(추가 개선). 16MB 이내 + 최고 bpb 동시 달성 시도.

## 베이스
Exp 115 (SupermaskTTT V3) + Exp 121 (NorMuon) + Exp 140 (Selective int5)

## 변경 사항
- NorMuon: Muon optimizer에 neuron-wise L2 normalization 추가
- EWQ: layers 0-2만 int5 양자화 (QAT + post-training)
- TTT: SupermaskTTT V3 (lr=0.005, epochs=3) 유지

## 예상 결과
- val_bpb (sliding): ~1.122 (NorMuon + EWQ 조합)
- val_bpb (TTT): ~1.121-1.122 (TTT 추가 개선)
- 크기: ~15.8-16.0MB (16MB 통과 목표)
- step_avg: ~95ms
