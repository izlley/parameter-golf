# Exp 168: MuonTTT + MoR Skip-Integrated

## 목적 (Purpose)
Exp 150 MoR(1.1271 bpb)의 핵심 버그 수정: layer 9의 skip connection이 완전히 무시되는 문제.
MoR iter1에 encoder 1 skip, iter2에 encoder 0 skip을 각각 주입하여 recursion 품질 향상.

## 베이스
Exp 162 (11L MuonTTT, val_bpb 1.1160, 18.1MB)

## 변경 사항
- **MoR**: Layer 8-9 공유 (blocks[9] = blocks[8])
- **Skip Integration**: iter1에 encoder 1 skip, iter2에 encoder 0 skip 주입
  - 기존 Exp 150: iter2에 skip 없음 (encoder 0 정보 버림)
  - 본 실험: 두 skip 모두 활용, recursion 차별화
- **DepthRouter**: dim→1, Gumbel-sigmoid routing
- **iter_scales**: per-iteration learnable scaling
- **TTT**: MuonLiteTTT + Supermask (Exp 162 그대로)

## 예상 결과
- val_bpb: ~1.1140-1.1180 (skip 통합으로 MoR 개선, TTT 시너지)
- artifact size: ~16.5-17.0MB (1 block 공유로 ~1.5MB 절감)
- step_avg: ~96-97ms (MoR 2회 반복 overhead)
