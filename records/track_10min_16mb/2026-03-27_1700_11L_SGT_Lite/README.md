# Exp 139: SGT Lite (Speed-Optimized Attention Looping)

## 목적 (Purpose)

Exp 117 SGT의 val_bpb 빠른 수렴 특성을 유지하면서 step_avg < 100ms 달성.
SGT V1 (126ms), V2 (102ms)의 속도 병목을 분석하고 근본적으로 제거.

## Base
Exp 129 (NorMuon, val_bpb 1.1183, step_avg 95ms, 6315 steps)

## 속도 병목 분석

1. **sgt_loop_attn** (+7ms/active_layer): attention 재실행 비용
2. **compute_head_entropy** (periodic +2-5ms spike): 명시적 [B,H,T,T] attention score 계산
3. **frozenset 변경 → recompile**: 성장 시 graph 재컴파일 (~수초)
4. **head_gate 곱셈**: gate 선택적 실행이 오히려 flash_attn 최적화를 방해

## 최적화 방안

### 1. Entropy 계산 완전 제거
- SGT V2에서 이미 layer 10만 선택이 최선이었음
- 고정 layer 선택 → compute_head_entropy 불필요 → periodic spike 제거

### 2. Head gate 제거 → 전체 head 루핑
- 2 head만 gate로 선택하는 것 < 8 head 전체를 flash_attn으로 실행
- head_gate 곱셈 자체가 ~0.5ms overhead
- flash_attn은 전체 head에서 최적화됨

### 3. 고정 구조 (recompile 0회)
- `_sgt_active_layers`를 처음부터 `frozenset({10})`으로 고정
- torch.compile recompile 전혀 없음

### 4. Learnable loop scale
- Loop attention 출력에 per-dim learnable scale 추가
- 고정 구조의 품질 손실 보상
- `loop_attn_scale = nn.Parameter(torch.zeros(dim))` (학습 시작 시 0 = no loop effect)

### 5. Loop warmup
- 처음 500 step: loop scale = 0 (base model과 동일하게 학습)
- 500+ step: loop scale 학습 시작
- 안정적 수렴 보장

## 예상 결과

- **step_avg**: ~98-100ms (95ms + ~3-5ms attention loop)
- **steps**: ~6,000+ (100ms 이내면 6000 달성)
- **val_bpb**: ~1.115-1.118 (NorMuon 대비 개선 가능)
- **artifact size**: ~17MB (NorMuon 동일)

## 실제 결과 (Actual Results)

(실행 후 기록)
