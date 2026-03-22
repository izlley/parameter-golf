# 13L Effective (Weight Sharing) + QAT

## 목적

11L의 깊이 이점(val_bpb 1.1362)은 입증되었으나 16MB 예산을 초과한다. Weight Sharing으로 **저장 파라미터는 10L과 동일하게 유지하면서 effective depth를 13L로 확장**하여, 크기 제약 없이 깊이 이점을 얻는다.

핵심 아이디어: 중간 블록(3,4,5)을 한 번 더 반복 실행하여 13 effective layers를 구성. 동일 블록이라도 입력이 다르므로(residual state가 다름) 다른 변환을 학습할 수 있다 (Universal Transformer 원리).

## 구현 방향

- **베이스**: 현재 최고 결과 (10L QAT, val_bpb 1.14239)
- **핵심 변경**:
  - `num_blocks=10` (저장되는 unique 블록 수, 기존과 동일)
  - `layer_pattern=[0,1,2,3,4,5,3,4,5,6,7,8,9]` (13 effective layers)
  - 중간 블록 3,4,5를 encoder와 decoder에서 각각 한 번씩 사용
  - U-Net skip: encoder 6층 + decoder 7층, skip_weights 6개
  - output projection scaling: `1/sqrt(2*13)` (effective depth 기준)
- **양자화/크기**: 저장 블록이 10개로 동일하므로 아티팩트 크기 ~15.77MB (기존과 동일)
- **학습 시간**: 블록 재사용으로 forward/backward가 30% 더 길어짐 예상 (~5000 steps)
- **그 외 설정**: int5 MLP + int6 Attn QAT, BigramHash 10240, pruning 5%, cosine warmdown 3500, SWA 0.35, eval stride=32

## Weight Sharing 패턴

```
Encoder (6 layers):  blocks[0] → [1] → [2] → [3] → [4] → [5]
                                                ↓ skip connections
Decoder (7 layers):  blocks[3] → [4] → [5] → [6] → [7] → [8] → [9]
                     (shared)   (shared) (shared)
```

- 블록 3,4,5: encoder와 decoder에서 각 1회씩 총 2회 실행
- 블록 0,1,2,6,7,8,9: 각 1회 실행
- 총 13회 블록 실행, 10개 unique 블록 저장

## torch.compile 호환성

- `layer_pattern`은 고정 Python list → tracing 시 상수로 해석
- 동일 `nn.Module`을 여러 번 호출하는 것은 표준 PyTorch 동작 (ALBERT, Universal Transformer)
- gradient는 공유 블록에 자동으로 합산됨

## 예상 결과

- **val_bpb**: 1.138~1.142
  - 11L unique (1.1362) 대비 약간 나쁠 수 있음 (공유 블록은 unique보다 적응력 제한)
  - 하지만 10L (1.1424) 대비 개선 기대 (깊이 증가 효과)
- **아티팩트 크기**: ~15.8MB (10L과 동일, skip_weights 1개 추가 ~2KB만 증가)
- **학습 시간**: step_avg ~120ms 예상 (13번 블록 실행 vs 10번), ~5000 steps
- **위험**: 중간
  - 공유 블록의 gradient가 2회분 합산 → 학습 불안정 가능성
  - step 수 감소(~5000 vs ~6400)가 깊이 이점을 상쇄할 수 있음
