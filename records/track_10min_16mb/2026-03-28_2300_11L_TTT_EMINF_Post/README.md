# Exp 163: EM-INF Post-TTT

## 목적
SupermaskTTT 평가 후 추가로 EM-INF (entropy minimization at inference) 적용. Logit 자체를 GD로 최적화하여 예측 엔트로피 최소화.

## 베이스
Exp 145 (NorMuon + SupermaskTTT V3, 1.1181 bpb)
Exp 127 (EM-INF 구현 참고)

## 변경 사항
1. eval_val_sliding_eminf 함수 추가 (Exp 127에서 복사)
2. TTT 평가 후 EM-INF 평가를 추가 실행
3. eminf_steps=5, eminf_lr=0.05 (Exp 127과 동일)
4. 별도 metric으로 eminf_loss, eminf_bpb 리포트

## 예상 결과
- TTT bpb: 1.1181 (변경 없음)
- EM-INF bpb: ~1.1170-1.1180 (TTT 대비 미세 개선 기대)
- 추가 eval 시간: ~80-100s

## 실제 결과
_(실험 후 기록)_
