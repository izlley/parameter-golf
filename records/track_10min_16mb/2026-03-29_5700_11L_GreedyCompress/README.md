# Exp 197: Greedy Iterative Compression (XSA AllLayers base)

## 목적 (Purpose)
Exp 189(XSA AllLayers, 1.1155 bpb, 17.9MB)를 16MB 이하로 정밀 압축.
일괄 int5 배정 대신, **텐서 단위로 가장 싼 것부터 하나씩** int5로 다운그레이드하여
정확히 16MB 이하에 도달하면 즉시 중단하는 Greedy Iterative Compression 방식.

## 베이스
Exp 189 (11L XSA AllLayers, TTT val_bpb 1.1155, 17.9MB)

## 변경 사항

### 1. Greedy Iterative Compression (`mixed_quantize_int6` 전면 개편)
- **Phase 1**: 모든 eligible 텐서를 int6으로 양자화 → 전체 compressed 크기 측정
- **Phase 2**: 각 텐서별 int5 다운그레이드 비용/이득 산출
  - cost = `(MSE_int5 - MSE_int6) × numel` (품질 손실)
  - benefit = `compressed_size(int6) - compressed_size(int5)` (실제 zstd 압축 크기 차이)
  - ratio = `cost / benefit` (낮을수록 "싸게" 압축 가능)
- **Phase 3**: ratio 오름차순으로 정렬, 가장 싼 텐서부터 int5로 전환
- **Phase 4**: 매 전환마다 전체 재압축하여 크기 측정 → **16MB 이하 도달 시 즉시 중단**

### 2. Int5 QAT (Heuristic)
- 학습 중 layers 0-5에 int5 QAT 적용 (`_ewq_int5 = True`, clip_range=15)
- Greedy 선택은 학습 후에 결정되므로, 학습 중에는 heuristic으로 early layers 마킹

### 3. Target Size Parameter
- `target_size=16,000,000`, `code_bytes` 전달하여 코드 크기도 고려

## 핵심 장점
- **정확히 필요한 만큼만 압축**: 불필요한 과도한 int5 적용 방지
- **텐서 단위 최적 선택**: 같은 레이어 내에서도 fc는 int5, proj는 int6 유지 가능
- **실측 기반**: zstd 압축 후 실제 크기로 판단 (추정이 아님)

## 예상 결과
- TTT val_bpb: ~1.1185-1.1210 (필요 최소한의 텐서만 int5, Exp 172의 +0.0067 대비 적은 손실)
- artifact size: ≤16,000,000B ✅ (greedy가 정확히 맞춤)
- step_avg: ~96-97ms (XSA all layers 포함)
- 양자화 시간: tensor당 compressed size 측정으로 ~30초 추가

## 실제 결과
(실행 후 기록)
