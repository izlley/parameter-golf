# Exp 166: MuonTTT + Pruning 12%

## 목적 (Purpose)
Exp 162 TTT_MuonOptimizer(1.1160 bpb, 18.1MB)의 크기를 16MB 이내로 축소.
Magnitude pruning 12%로 작은 weight를 0으로 만들어 zstd 압축률 향상.

## 베이스
Exp 162 (11L TTT_MuonOptimizer, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- EMA 적용 후, 양자화 전에 magnitude pruning 12% 적용
- 2D weight 중 크기가 하위 12%인 값을 0으로 설정
- 0 값이 많아지면 zstd 압축률 향상 → 크기 절감
- TTT 부분은 Exp 162와 완전 동일

## 예상 결과
- TTT val_bpb: ~1.1165-1.1190 (pruning에 의한 소폭 품질 손실)
- artifact size: ~16.5-17.5MB (Exp 130에서 ~0.4MB 절감 확인, 충분하지 않을 수 있음)
- step_avg: ~95ms (변경 없음)

## 실제 결과
(실행 후 기록)
