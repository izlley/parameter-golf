# Exp 199: PR#1019 + GatedAttn + TTT + Size Fix

## 목적
Exp 196 (sliding bpb 1.1182, 최고 성능)이 672KB 크기 초과로 제출 불가였던 것을 수정.
원인: TARGET_MB=15.9가 MiB 단위(16.67MB decimal)로 계산되어 16MB(16,000,000B) 초과.

## 베이스
Exp 196 (PR#1019 + GatedAttn + SupermaskTTT, sliding 1.1182, 16,672,143B)

## 변경 사항
- `TARGET_MB` 기본값: 15.9 → **15.15** (15.15 MiB ≈ 15.88MB decimal, 16MB 이내)
- selective pruning이 더 공격적으로 ±1 값을 제거하여 크기 축소
- 학습/모델 코드 변경 없음 — pruning 강도만 변경

## 예상 결과
- val_bpb: ~1.1182 (Exp 196과 동일 학습, pruning 약간 증가로 미미한 bpb 손실)
- artifact size: ≤16,000,000B (15.88MB 타겟)
- step_avg: 94.29ms (동일)

## 실제 결과
_(실행 후 기록)_
