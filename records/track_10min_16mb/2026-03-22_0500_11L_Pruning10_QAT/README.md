# 11L + Pruning 10% + QAT

## 목적

11L은 val_bpb 1.1362 (전체 최고)를 달성했으나 크기 초과(19MB, 비트패킹 역효과)로 실패했다. 이전 시도에서 BigramHash 축소(10240→8192)로 크기를 맞추면 val_bpb 손해(-0.002)가 깊이 이점을 상쇄했다.

이번 실험: **BigramHash 10240을 유지하면서 pruning을 5%→10%로 올려** 크기를 절약한다. Pruning은 가중치를 0으로 만들어 zstd 압축률을 높이므로, BigramHash 축소보다 val_bpb 손해가 적을 것으로 기대한다.

## 구현 방향

- **베이스**: 현재 최고 결과 (10L QAT, val_bpb 1.14239)의 11L 버전
- **핵심 변경**:
  - `num_layers`: 10 → **11**
  - `FP16_KEEP`: blocks.8.attn.c_k → **blocks.9.attn.c_k** (끝에서 두 번째 레이어)
  - Pruning: **5% → 10%** (magnitude pruning quantile 0.05 → 0.10)
  - BigramHash: **10240 유지** (축소 없음)
- **그 외 설정**: int5 MLP + int6 Attn QAT(STE), cosine warmdown 3500, SWA 0.35, Muon WD=0.04, eval stride=32

## 크기 추정

- 10L SOTA: 15.77MB
- 11L 추가 블록 비용: ~700-800KB (int5/int6 양자화 + zstd 후)
- Pruning 10% 절약: ~300-500KB (더 많은 0 → zstd 압축률 향상)
- **예상 크기: ~16.0-16.2MB** (빠듯하지만 16MB 이내 가능)

## 예상 결과

- **val_bpb**: ~1.138~1.142
  - 11L 깊이 이점: -0.002~0.003 bpb
  - Pruning 10% 손해: -0.001~0.002 bpb (5%보다 더 많은 가중치 제거)
  - 순 효과: -0.001~0.002 bpb 개선 기대
- **아티팩트 크기**: ~16.0MB (빠듯)
- **학습 시간**: step_avg ~95ms (11L, 10L 대비 +2-3ms)
- **위험**: 중
  - 크기가 빠듯하여 16MB 초과 가능성 있음
  - Pruning 10%의 val_bpb 영향이 클 수 있음

## 실제 결과

(학습 후 기록 예정)
