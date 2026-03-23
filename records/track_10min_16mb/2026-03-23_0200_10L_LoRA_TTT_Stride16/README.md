# 10L LoRA TTT + Stride 16

## 목적

현재 SOTA(10L, relu^2 MLP, val_bpb 1.14239)의 **평가 시** LoRA 기반 Test-Time Training(TTT)을 적용하여, 모델이 각 평가 배치에 동적으로 적응하도록 한다. 평가에 별도 10분 예산이 있으므로 이를 최대한 활용한다.

## 핵심 아이디어

TTT(Test-Time Training)는 추론 시 입력 데이터에 대해 모델을 소량 적응시키는 기법이다:
1. 각 평가 배치마다 소규모 LoRA 어댑터를 주입
2. 해당 배치의 next-token prediction loss로 LoRA 파라미터만 몇 스텝 학습
3. 적응된 모델로 평가 후, LoRA 제거하고 원래 모델 복원
4. 다음 배치에서 반복

모델 가중치 자체는 변경하지 않으므로 **아티팩트 크기에 영향 없음**.

## 구현 방향

- **베이스**: 현재 SOTA (10L QAT, val_bpb 1.14239, 15.77MB)
- **eval_stride**: 32 → **16** (각 토큰이 더 많은 컨텍스트로 평가됨)
- **LoRA TTT 설정**:
  - `ttt_rank`: 4 (LoRA 랭크)
  - `ttt_lr`: 1e-4 (Adam 학습률)
  - `ttt_steps`: 3 (배치당 적응 스텝 수)
  - `ttt_target_modules`: c_q, c_v (Query, Value projection에 LoRA 주입)
- **그 외 설정**: SOTA와 동일 (int5/int6 QAT, BigramHash 10240, pruning 5%, SWA 0.35)

## LoRA TTT 동작 방식

```
각 eval 배치에 대해:
  1. LoRA 주입: c_q, c_v에 rank-4 어댑터 추가
  2. 적응 학습: 3 스텝 × Adam(lr=1e-4), 해당 배치로 self-supervised
  3. 적응 평가: 학습된 LoRA로 logits 계산 → NLL 기록
  4. LoRA 제거: 원래 모델 복원
```

## stride=16의 의미

- stride=32 (SOTA): 각 윈도우에서 마지막 32 토큰만 점수화
- stride=16: 각 윈도우에서 마지막 16 토큰만 점수화 → 윈도우 수 2배 → 더 많은 컨텍스트
- 평가 시간 ~2배 증가, but 별도 10분 예산이므로 충분

## 예상 결과

- **val_bpb**: ~1.138~1.142
  - stride 16: -0.001 bpb (더 많은 컨텍스트로 평가)
  - LoRA TTT: -0.001~0.003 bpb (배치별 적응 효과)
- **아티팩트 크기**: ~15.8MB (SOTA와 동일, LoRA는 eval-only)
- **학습 시간**: SOTA와 동일 (~93ms/step)
- **평가 시간**: ~8-10분 (stride 16 + TTT 오버헤드)
- **위험**: 중
  - TTT 스텝이 3이면 적응이 부족할 수 있음 (underfitting)
  - 배치 내 윈도우들이 서로 다른 문서 → cross-contamination 가능
  - LoRA 주입/제거 오버헤드로 평가 시간 초과 위험

## 실제 결과

(학습 후 기록 예정)
