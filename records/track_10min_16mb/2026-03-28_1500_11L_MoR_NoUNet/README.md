# Exp 155: MoR (Mixture-of-Recursions) + U-Net 제거

## 목적 (Purpose)

Exp 150 MoR(1.1271 bpb)의 개선판. U-Net skip connection이 MoR recursion과 충돌하는 문제 해결.

**핵심 가설**:
1. U-Net skip이 공유 레이어의 입력 분포를 왜곡하여 recursion 품질 저하 유발
2. U-Net 제거 후 순수 residual connection으로 MoR이 더 깨끗한 recursion 수행
3. 공유 위치를 중앙(5-6)으로 변경하여 양쪽 레이어에서 균등한 정보 흐름

**Exp 150 대비 변경점**:
- U-Net (encoder/decoder + skip_weights) 완전 제거 → 단순 sequential forward
- MoR 공유 레이어: 8-9 → 5-6 (중앙 레이어, 더 범용적)
- Router hidden dim: 64 → 32 (파라미터 절감)

## 베이스
Exp 121 (11L NorMuon, val_bpb 1.1183, 17.16MB)

## 변경 사항
- **U-Net 제거**: `skip_weights`, `num_encoder_layers`, `num_decoder_layers` 모두 삭제
- **Sequential forward**: 모든 레이어를 순차 실행 (encoder/decoder 구분 없음)
- **MoR**: Layer 5-6을 하나의 공유 블록으로 교체 (최대 2회 반복)
  - DepthRouter: dim→32→1, Gumbel-sigmoid routing
  - iter_scales: per-iteration learnable scaling
  - torch.compile 호환: 항상 2회 실행 + mask blend
- **파라미터**: skip_weights 제거(~2.5K), router 추가(~16.4K), 1 block 공유(~2.4M 절감)

## 예상 결과
- val_bpb: ~1.1240-1.1270 (U-Net 제거 + 중앙 공유로 Exp 150 개선 기대)
- artifact size: ~15.5-16.0MB (1 block 절감)
- step_avg: ~95-96ms (U-Net overhead 제거, MoR 2회 반복)

## 실제 결과
(실행 후 기록)
