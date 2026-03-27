# Exp 138: Triton-Fused Mamba2 SSD Kernel

## 목적 (Purpose)

`_ssd_chunked`의 핵심 연산을 Triton kernel로 퓨전하여 Mamba2의 속도를 개선.
목표: 99ms → ~93-95ms (kernel fusion으로 4-6ms 절약).

**핵심 가설**:
1. `_segsum` + intra-chunk 연산을 Triton kernel로 퓨전 → 메모리 대역폭 절약
2. inter-chunk recurrence는 별도 Triton kernel
3. `torch.library.custom_op`으로 래핑하여 torch.compile(fullgraph=True) 호환

## 구현 방향 (Implementation Approach)

**베이스**: Exp 134 (Mamba2Light + SharedMLP)

**Triton 최적화 포인트**:
1. `_segsum`: O(T^2) cumsum masking → tiled Triton kernel
2. Intra-chunk: `torch.einsum` 4개 → fused Triton matmul
3. Inter-chunk recurrence: sequential scan kernel
4. 메모리: intermediate tensor 제거 (kernel 내부 처리)

**구현 전략**:
- Forward kernel: `triton_ssd_fwd` (intra-chunk attention + segsum fusion)
- Backward: torch.autograd로 자동 미분 (forward는 Triton, backward는 PyTorch)
- `torch.library.custom_op` + `torch.library.impl_abstract`로 torch.compile 호환

**Fallback**:
- Triton 미설치/컴파일 실패 시 순수 PyTorch 구현으로 자동 fallback
- 속도만 차이, 수학적 결과는 동일

## 예상 결과 (Expected Results)

- **step time**: ~93-95ms (kernel fusion으로 4-6ms 절약)
- **val_bpb**: Exp 134와 동일 (수학적으로 같은 연산)
- **artifact size**: Exp 134와 동일

## 실제 결과 (Actual Results)

(실행 후 기록)
