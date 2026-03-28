# Exp 169: MuonTTT + MoR Layer 9-10

## 목적 (Purpose)
MoR을 skip connection과 충돌 없는 위치(layer 9-10)에 배치.
Layer 9은 마지막 skip(encoder 0)을 정상 수신, layer 10은 skip 없이 순수 refinement recursion.

## 베이스
Exp 162 (11L MuonTTT, val_bpb 1.1160, 18.1MB)

## 변경 사항
- **MoR**: Layer 9-10 공유 (blocks[10] = blocks[9])
- **Skip 충돌 없음**: 모든 5개 encoder skip이 정상 소비됨
  - Layer 9 (iter1): encoder 0 skip 수신
  - Layer 10 (iter2): skip 없음, iter1 출력의 순수 recursion
- **DepthRouter**: dim→1, Gumbel-sigmoid routing
- **iter_scales**: per-iteration learnable scaling
- **TTT**: MuonLiteTTT + Supermask (Exp 162 그대로)

## 예상 결과
- val_bpb: ~1.1130-1.1170 (skip 충돌 없는 깔끔한 MoR)
- artifact size: ~16.5-17.0MB (1 block 공유로 ~1.5MB 절감)
- step_avg: ~96-97ms (MoR 2회 반복 overhead)
