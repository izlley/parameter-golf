# Exp 157: TTT Input-Dependent Dynamic Masking

## 목적
고정 mask (static `sigmoid(mask_score)` per channel) 대신 입력 의존적 dynamic mask generator를 사용하여 TTT 단계의 표현력을 증가시킨다. 입력 시퀀스의 통계에 따라 mask가 적응적으로 변하므로 다양한 텍스트 패턴에 대해 더 정밀한 마스킹이 가능하다.

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3)

## 변경 사항
- **DynamicMaskGenerator 클래스 추가**: `input_dim -> bottleneck(64) -> output_dim` 구조의 경량 2-layer MLP
  - 입력: 시퀀스 mean pooling `(batch, seq_len, dim) -> (batch, dim)`
  - 출력: `(batch, output_dim)` mask logits (sigmoid 적용 전)
  - 초기화: down/up weight=0, up.bias=3.0 (기존 static mask와 동일한 시작점)
- **Per-block 구조**: MLP용 generator + Attn용 generator (각 블록당 2개)
- **Static bias 유지**: 기존 additive bias term은 static으로 유지
- **Phase 1 변경**: `torch.inference_mode()` -> `torch.no_grad()` (generator forward pass 호환)
- **Optimizer**: 모든 generator parameter + static bias를 SGD로 학습

## 예상 결과
- val_bpb: -0.002 ~ -0.005 개선 (입력 적응형 마스킹으로 인한 향상)
- TTT 시간: +30% (generator forward pass 추가)
- Artifact size: 변화 없음 (generator는 TTT 시에만 사용, 저장하지 않음)

## 실제 결과
_(예정)_
