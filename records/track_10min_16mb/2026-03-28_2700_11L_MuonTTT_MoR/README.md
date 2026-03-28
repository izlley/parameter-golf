# Exp 167: MuonTTT + MoR (Layer 8-9 공유)

## 목적 (Purpose)
Exp 162 TTT_MuonOptimizer(1.1160 bpb, 18.1MB)의 크기를 16MB 이내로 축소.
MoR(Mixture-of-Recursions)로 Layer 8-9를 공유하여 ~1.5MB 파라미터 절감.

## 베이스
Exp 162 (11L TTT_MuonOptimizer, TTT val_bpb 1.1160, 18.1MB)

## 변경 사항
- Layer 8-9를 하나의 공유 블록으로 교체 (U-Net 유지)
- DepthRouter: per-token adaptive depth (Gumbel-sigmoid)
- iter_scales: per-iteration learnable scaling
- torch.compile 호환: 항상 2회 실행 + mask blend
- TTT 부분은 Exp 162와 완전 동일

## 예상 결과
- TTT val_bpb: ~1.1170-1.1200 (1 block 공유 손실)
- artifact size: ~16.0-16.5MB (1 block 절감 ~1.5MB)
- step_avg: ~96ms (MoR 2회 반복 오버헤드)

## 실제 결과
(실행 후 기록)
