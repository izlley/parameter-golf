# Exp 189: NorMuon + XSA All Layers

## 목적
XSA (Exclusive Self-Attention)를 현재 last 4 layers에서 all 11 layers로 확장하여 bpb를 개선한다.
abaybektursun PR#1019에서 XSA all layers로 TTT 없이도 1.1147 bpb를 달성한 것을 참고.

## 베이스
Exp 162 (11L_TTT_MuonOptimizer / NorMuon + MuonLiteTTT, 1.1160 bpb, 18.1MB)

## 변경 사항
1. `xsa_last_n` 기본값: 4 -> **11** (모든 레이어에 XSA 적용)
2. 학습/양자화/TTT 등 나머지 동일

## 배경
- XSA는 attention output에서 self-value projection을 빼서 각 토큰이 자기 자신의 value에 과적합하는 것을 방지
- Last 4 layers만 적용 시 encoder layers는 XSA 없이 학습
- abaybektursun은 all layers XSA로 전환 후 TTT도 제거 가능했음
- XSA는 연산량 증가 미미 (projection subtraction만 추가)

## 예상 결과
- val_bpb: ~1.1140-1.1170 (last 4 대비 개선 기대)
- step_avg: ~95ms (XSA 오버헤드 미미)
- artifact size: 18.1MB (동일, 모델 구조 변경 없음)
- TTT bpb: 추가 개선 가능

## 실제 결과
_(실행 후 기록)_
