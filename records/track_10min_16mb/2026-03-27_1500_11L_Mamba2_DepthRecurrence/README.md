# Exp 137: SSM + Depth Recurrence Hybrid (Mamba2 x2 Loop)

## 목적 (Purpose)

Mamba2 layer를 2회 반복 실행하여 effective depth 증가.
SSM의 sequential state 업데이트는 attention보다 양자화 오류 복합이 적을 수 있다는 가설 검증.

**핵심 가설**:
1. 9L Attention (0-8) + Mamba2 2회 반복 (position 9, 10) = effective 11L
2. Mamba2 shared weight → 사이즈 절감 (MLP 1개 + SSM 1개 절약)
3. Iteration embedding으로 반복 간 차별화
4. SSM state가 반복 시 attention보다 graceful하게 작동

## 구현 방향 (Implementation Approach)

**베이스**: Exp 134 (Mamba2Light + SharedMLP)

**변경 사항**:
- 10개 Attention Block (layers 0-9) + 1개 Mamba2Block (layer 10)
- Layer 10의 Mamba2Block을 2회 반복 실행 (positions 10, 11 역할)
- `mamba_loop_count=2`: Mamba2 layer 반복 횟수
- Iteration embedding: `iter_embed = nn.Parameter(torch.zeros(loop_count, dim))` per loop iteration
- Forward: `x = x + iter_embed[i]` before each Mamba2 pass
- 나머지 9개 attention layer는 flat (변경 없음)

**사이즈 효과**:
- Mamba2Block 1개만 저장 (반복이므로)
- Attention Block 10개 + Mamba2Block 1개 ≈ flat 11L 대비 Mamba2 1개 절약

## 예상 결과 (Expected Results)

- **step time**: ~98-100ms (Mamba2 x2 실행)
- **val_bpb**: ~1.1200-1.1240
- **artifact size**: ~16.0-16.5MB (Mamba2 1개 절약)

## 실제 결과 (Actual Results)

(실행 후 기록)
