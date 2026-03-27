# Exp 140: NorMuon + Selective int5 (layers 0-2 only)

## 목적
Exp 131(NorMuon+EWQ, layers 0-5 int5)의 bpb 손실(+0.011)을 줄이기 위해 int5 적용 범위를 layers 0-2로 축소.

## 베이스
Exp 131 (NorMuon+EWQ, val_bpb 1.1296, 15,554,069B)

## 변경 사항
- EWQ int5 적용 범위: layers 0-5 → **layers 0-2** (3개 레이어만)
- QAT에서도 동일하게 layers 0-2만 int5 fake quantization

## 예상 결과
- val_bpb: ~1.122 (int5 범위 축소로 bpb 손실 감소)
- 크기: ~15.8-16.0MB (16MB 통과 목표)
- step_avg: ~95ms (변경 없음)

## 실제 결과
_(예정)_
