# Exp 148: Hadamard QAT — QuEST-style int4 재도전

## 목적 (Purpose)
QuEST 논문의 Hadamard rotation + MSE-optimal clipping을 적용하여 MLP weight를 int4로 양자화.
기존 int4 시도(Exp 22)는 "16 levels 부족" 실패. Hadamard rotation이 weight outlier를 분산시켜
uniform quantization에 최적화된 분포를 만드는 것이 핵심 차이.

## 베이스
Exp 121 (11L NorMuon, val_bpb 1.1183, 17.16MB)

## 변경 사항
- **Hadamard Rotation**: MLP fc/proj의 CastedLinear forward에 Hadamard matrix 적용
  - W_eff = W @ H (offline rotation), x_rot = x @ H^T (online rotation)
  - Hadamard은 직교 → 정보 손실 없이 outlier 분산
- **QAT int4**: MLP weight에 int4 QAT 적용 (clip_range=7, 4-bit signed)
  - Attention weight는 기존 int6 QAT 유지
- **MSE-optimal clipping**: min-max 대신 MSE 최소화 기준 clipping range 선택
- **저장**: MLP int4 + Attention int6 mixed quantization

## 예상 결과
- val_bpb: ~1.1200-1.1230 (int4 손실 있지만 Hadamard로 최소화)
- artifact size: ~12-13MB (MLP weight 33% 추가 축소)
- step_avg: ~96-97ms (Hadamard 연산 약간 추가)

## 실제 결과
(실행 후 기록)
