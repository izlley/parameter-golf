# Exp 149: Relaxed Recursive Transformer — LoRA-Adapted Weight Sharing

## 목적 (Purpose)
ICLR 2025 "Relaxed Recursive Transformers" 논문 적용.
레이어 weight를 공유하되, 각 반복에 tiny LoRA adapter를 추가하여 표현력 유지.
기존 SharedFFN(6회 실패)과 depth recurrence(4회 실패)의 문제를 LoRA relaxation으로 해결.

## 베이스
Exp 121 (11L NorMuon, val_bpb 1.1183, 17.16MB)

## 변경 사항
- **Layer Sharing**: 11L 중 Layer 4-5를 공유, Layer 6-7을 공유 (2쌍 공유)
  - 실효 11L 동작하지만 고유 파라미터는 ~9L 수준
- **LoRA Adapter**: 공유되는 각 "복제" 레이어에 rank-4 LoRA 추가
  - Attention의 c_q, c_v에 LoRA: delta_W = A @ B (A: dim×4, B: 4×dim)
  - MLP의 fc에 LoRA: delta_W = A @ B (A: dim×4, B: 4×hidden)
  - LoRA 파라미터: ~4 × (512×4 + 4×512) × 2쌍 ≈ 32K params (무시 가능)
- **U-Net Skip**: 기존 encoder-decoder 구조 유지

## 예상 결과
- val_bpb: ~1.1200-1.1220 (LoRA가 공유 손실 보상)
- artifact size: ~14.5-15MB (2L 분량 파라미터 절감)
- step_avg: ~95ms (LoRA overhead 무시 가능)

## 실제 결과
(실행 후 기록)
