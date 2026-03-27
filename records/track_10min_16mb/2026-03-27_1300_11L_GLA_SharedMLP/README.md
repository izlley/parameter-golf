# Exp 135: Gated Linear Attention (Layer 10) + Shared MLP

## 목적 (Purpose)

Mamba2 대신 GLA(Gated Linear Attention)로 대체하여 더 빠른 SSM을 달성.
GLA는 conv1d가 없고 구조가 단순하여 torch.compile 최적화에 더 유리할 것으로 예상.

**핵심 가설**:
1. GLA가 Mamba2보다 ~2-3ms 빠름 (conv1d 없음, 더 단순한 recurrence)
2. chunk-wise parallel training으로 causal attention과 유사한 품질 달성
3. Layer 10의 MLP를 Layer 5와 공유하여 사이즈 절감

## 구현 방향 (Implementation Approach)

**베이스**: Exp 134 (Mamba2Light+SharedMLP) 에서 Mamba2 → GLA 교체

**GLA 아키텍처**:
- Input projection: d_model -> (Q, K, V, gate) = d_model*4
- Q, K: head_dim per head (standard)
- V: head_dim per head
- Gate: sigmoid gate per head for state decay
- Chunk-wise computation: intra-chunk은 행렬곱, inter-chunk은 recurrence
- State: h_t = gate_t * h_{t-1} + k_t^T * v_t (gated outer product)
- Output: y_t = (q_t @ h_t)
- No conv1d, no special init (simpler than Mamba2)

**주요 차이점 vs Mamba2**:
- Conv1d 제거 → ~1ms 절약
- 단순한 게이트 구조 → torch.compile 최적화 용이
- State 업데이트가 outer product → d_state 불필요 (head_dim이 곧 state dim)

## 예상 결과 (Expected Results)

- **step time**: ~94-96ms (Mamba2 대비 ~1-2ms 빠름)
- **val_bpb**: ~1.1200-1.1230 (Mamba2 대비 약간 열위 가능)
- **artifact size**: ~16.0-16.5MB

## 실제 결과 (Actual Results)

(실행 후 기록)
