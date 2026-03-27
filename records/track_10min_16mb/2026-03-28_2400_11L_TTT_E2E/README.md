# Exp 164: E2E TTT (End-to-End Test-Time Training)

## 목적
E2E TTT 논문(arXiv:2512.23675) 아이디어 — 학습 중 TTT를 시뮬레이션하여 모델 초기화를 TTT에 최적화. Meta-learning으로 "적응하기 좋은 모델"을 학습.

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)

## 변경 사항
1. e2e_ttt_rehearsal() 함수: 학습 중 임시 mask 생성 → inner loop 학습 → meta loss 계산
2. 매 500 step마다 rehearsal 실행 (e2e_ttt_interval=500)
3. inner_steps=2, inner_lr=0.01
4. meta loss weight = 0.1 (정규 학습 loss 대비)
5. create_graph=True로 mask gradient를 통한 model weight 업데이트
6. raw model (base_model) 사용으로 torch.compile monkey-patching 충돌 회피

## 예상 결과
- TTT bpb: ~1.1150-1.1170 (TTT에 최적화된 초기화)
- step_avg: +5-10ms (rehearsal 오버헤드, 매 500 step마다만 실행)
- 총 steps 감소 가능 → net 효과는 불확실

## 실제 결과
_(실험 후 기록)_
