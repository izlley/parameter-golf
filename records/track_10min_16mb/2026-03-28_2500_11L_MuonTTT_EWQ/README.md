# Exp 165: MuonTTT + EWQ (int5 양자화)

## 목적 (Purpose)
Exp 162 TTT_MuonOptimizer(1.1160 bpb, 18.1MB)의 크기를 16MB 이내로 축소.
EWQ(Element-Wise Quantization)로 early layers(0-5)를 int5로 양자화하여 ~1.6MB 절감.

## 베이스
Exp 162 (11L TTT_MuonOptimizer, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- Early layers(0-5): int5 양자화 (clip_range=15, 5-bit)
- Late layers(6-10): 기존 int6 유지 (clip_range=31, 6-bit)
- CastedLinear QAT에 int5 분기 추가
- TTT 부분은 Exp 162와 완전 동일 (MuonLiteTTT, lr=0.002, epochs=3)

## 예상 결과
- TTT val_bpb: ~1.1165-1.1185 (int5 양자화 손실 소폭)
- artifact size: ~15.5-16.0MB (Exp 131에서 ~1.6MB 절감 확인)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
