# Exp 198: PR#1019 Baseline (abaybektursun 원본)

## 목적
abaybektursun PR#1019 코드를 변경 없이 그대로 실행하여 우리 서버에서의 baseline 성능을 측정한다.
이후 실험(192-196)과의 정확한 비교 기준점.

## 베이스
abaybektursun PR#1019 원본 (리더보드 pending 1.1147 bpb)

## 변경 사항
- flash_attn import 경로만 try/except fallback 추가 (환경 호환성)
- 그 외 코드 변경 없음

## 측정 항목
- step_avg (PR#1019 원본은 ~83ms 보고)
- val_bpb, sliding_window bpb
- artifact size (GPTQ + LZMA + selective pruning)
- 총 학습 steps

## 실제 결과
_(실행 후 기록)_
