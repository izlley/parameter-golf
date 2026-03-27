# Exp 134: Lightweight Mamba2 (Layer 10) + Shared MLP

## 목적 (Purpose)

SSM(Mamba2)의 속도를 개선하면서 MLP 공유로 아티팩트 사이즈 절감을 동시에 달성.

**핵심 가설**:
1. Mamba2 하이퍼파라미터 경량화(d_state=32, chunk_size=128, d_conv=2)로 99ms → ~95ms 달성
2. Layer 10의 MLP를 Layer 5와 공유하여 ~1.2MB 사이즈 절감
3. SSM + Attention hybrid가 pure attention보다 높은 BPB를 달성

## 구현 방향 (Implementation Approach)

**베이스**: Exp 121 (NorMuon, 1.1183 bpb, 95ms/step) + Exp 83 (Mamba2 pure PyTorch)

**아키텍처**:
- Layers 0-9: GatedAttn Block (기존 NorMuon 동일)
- Layer 10: Mamba2Block (SSM) + Shared MLP (Layer 5의 MLP 재사용)
  - d_state=32 (기존 64 → 절반, 메모리/연산 절약)
  - chunk_size=128 (기존 64 → 2배, inter-chunk recurrence 절반)
  - d_conv=2 (기존 4 → 절반, conv kernel 축소)
  - expand=2 유지 (축소 시 용량 손실 큼)
- Shared MLP adapter: per-iteration scale vector (512 params)

**최적화 기법** (NorMuon 기반):
- NorMuon optimizer (neuron-wise L2 normalization)
- GatedAttn PerHead (bias=4.0)
- LeakyReLU(0.5)², MLP 3x
- XSA last 4 layers (layers 7,8,9 + Mamba2 layer 10은 XSA 미적용)
- BigramHash 8192, Value Embedding layers 9,10
- Partial RoPE 16/64
- Late QAT 0.15, EMA 0.997
- U-Net skip connections

**Mamba2 속도 최적화 포인트**:
- d_state 64→32: _segsum 연산의 N 차원 절반 → einsum 비용 절감
- chunk_size 64→128: chunk 수 절반 → inter-chunk recurrence 오버헤드 50% 감소
- d_conv 4→2: conv1d kernel 절반 → 미미한 절감이지만 누적 효과

## 예상 결과 (Expected Results)

- **step time**: ~95-97ms (목표: 6000+ steps in 10min)
- **val_bpb**: ~1.1190-1.1210 (NorMuon 수준 유지 + Mamba2 이점)
- **artifact size**: ~16.0-16.5MB (MLP 공유로 ~1.2MB 절감)
  - 추가 사이즈 최적화 필요 시 EWQ(int5 early layers) 적용 검토

## 실제 결과 (Actual Results)

(실행 후 기록)
