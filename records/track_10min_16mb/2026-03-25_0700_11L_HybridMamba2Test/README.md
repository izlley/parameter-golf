# Exp 83: Hybrid SSM+Attention Compile Test (Mamba2)

## 목적
순수 PyTorch Mamba2(SSD) 구현의 torch.compile(fullgraph=True) 호환성 검증.
Layer 0,1,2를 Mamba2로 교체, 나머지 8개는 기존 Attention 유지.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)

## 변경 사항

| 파라미터 | Exp 73 | Exp 83 | 근거 |
|----------|--------|--------|------|
| Layer 0,1,2 | Attention | **Mamba2 (SSD)** | 얕은 레이어는 SSM으로 대체 가능 |
| Layer 3-10 | Attention | Attention (유지) | 깊은 레이어는 attention 유지 |
| Mamba d_state | - | 64 | SSM hidden state 차원 |
| Mamba expand | - | 2 | d_inner = 2 × d_model |
| Mamba chunk_size | - | 64 | SSD chunked computation |
| Mamba d_conv | - | 4 | Depthwise conv1d kernel size |

## Mamba2 SSD 구현
- 순수 PyTorch (einops, Triton, custom CUDA 없음)
- `_segsum` → `_ssd_chunked` 4-step 알고리즘:
  1. Intra-chunk diagonal blocks (1-semiseparable matrix)
  2. Per-chunk state computation (B terms + decay)
  3. Inter-chunk state recurrence (A terms)
  4. State-to-output conversion (C terms + decay)
- `Mamba2Layer`: in_proj → conv1d → SSD → gated RMSNorm → out_proj
- `Mamba2Block`: Mamba2Layer + MLP (기존 Block과 동일 인터페이스)

## 검증 항목
1. ✅ torch.compile(fullgraph=True) 호환성
2. ✅ DDP 호환성
3. ✅ 학습 수렴 여부
4. ✅ step_avg 비교 (attention vs Mamba2 레이어 속도)
5. ✅ int6 양자화 + zstd 압축 호환성

## 환경 변수
```bash
MAMBA_LAYERS="0,1,2"      # Mamba2로 교체할 레이어 인덱스 (기본: "0,1,2")
MAMBA_D_STATE=64           # SSM state dimension
MAMBA_D_CONV=4             # conv1d kernel size
MAMBA_EXPAND=2             # expansion factor
MAMBA_CHUNK_SIZE=64        # SSD chunk size
```

## 예상 결과
- **compile 호환**: 미검증 (이것이 핵심 테스트)
- val_bpb: 성능보다 호환성 검증이 목적
- 위험: 높음 (compile 실패 가능, 학습 불안정 가능)

## 실제 결과
(학습 후 기록 예정)
