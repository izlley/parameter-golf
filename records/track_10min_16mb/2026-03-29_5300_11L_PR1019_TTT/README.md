# Experiment 193: PR#1019 Base + SupermaskTTT

## 목적
abaybektursun의 PR#1019 코드(Parameter Banking 아키텍처 + GPTQ int6 + selective pruning)를 베이스로 SupermaskTTT 평가를 추가하여 test-time training으로 val_bpb를 추가 개선.

## 베이스
- PR#1019 (abaybektursun): Parameter Banking 아키텍처, GPTQ int6 양자화, selective +-1 pruning

## 변경 사항
1. **TTT 하이퍼파라미터 추가**: `ttt_enabled`, `ttt_lr=0.002`, `ttt_epochs=3`, `ttt_chunk_tokens=32768`, `ttt_momentum=0.9`, `ttt_batch_seqs=32`, `ttt_grad_clip=1.0`
2. **MuonLiteTTT 옵티마이저**: Sign-based updates with Nesterov momentum (경량 TTT 학습용)
3. **SupermaskTTT 평가 함수 (`eval_val_sliding_ttt`)**: Parameter Banking 아키텍처에 맞게 적응:
   - MLP monkey-patch: `forward(x, up_w, down_w)` 시그니처에 맞게 hidden activation에 sigmoid mask + bias 적용
   - Attn monkey-patch: `forward(x, q_w, k_w, v_w, out_w, v_embed, v0)` 시그니처에 맞게 출력에 sigmoid mask + bias 적용
   - Bank 텐서에서 차원 정보 추출: `mlp_up_bank.shape[1]` = hidden_dim, `qo_bank.shape[1]` = model_dim
4. **Score-first legality**: 각 chunk를 먼저 inference_mode로 스코어링한 후 mask 학습
5. **LR 스케줄**: warmup 5% -> constant 65% -> cosine decay 30%

## 예상 결과
- `final_int6_sliding_window_bpb`: PR#1019 베이스와 동일
- `legal_ttt_bpb`: 슬라이딩 윈도우 대비 0.005-0.015 BPB 개선 예상
- step time: 베이스와 동일 (TTT는 평가 시에만 동작)
- artifact size: 16MB 이하

## 실제 결과
_(실험 실행 후 기입)_
