# Exp 183: QuaRot-Style Hadamard Rotation + SensitivityEWQ

## 목적 (Purpose)
양자화 전 weight matrix에 Hadamard rotation을 적용하여 outlier를 균등 분산.
Row-wise 분포가 더 uniform해지면 per-row 양자화 에러가 크게 감소.

## 베이스
Exp 172 (11L MuonTTT SensitivityEWQ, TTT val_bpb 1.1227, 15.9MB)

## 변경 사항
- **Hadamard Rotation**: 각 2D weight tensor의 열(column)에 Hadamard 변환 적용
  - 양자화: W_rot = W @ H → quantize(W_rot) → 저장
  - 역양자화: dequant(Q) @ H^T → W_approx (H^T = H for normalized Hadamard)
  - H는 deterministic seed로 생성 → artifact에 저장 불필요
  - power-of-2 크기만 적용 (512, 1536 등), 비 p2 크기는 random QR orthogonal
- **근거**: QuaRot (NeurIPS 2024) — LLaMA-2-70B에서 int4 품질 갭 43% 감소
- **기존 SensitivityEWQ 유지**: Hadamard rotation + sensitivity-based int5/int6 분배

## 예상 결과
- TTT val_bpb: ~1.1195-1.1215 (EWQ 비용 +0.0067 → +0.003~0.005로 감소)
- artifact size: ~15.9MB ✅ (변동 없음 — H는 재생성)
- step_avg: ~95ms (변경 없음)
- 양자화/역양자화 시간: H 곱셈 ~1-2초 추가

## 실제 결과
(실행 후 기록)
