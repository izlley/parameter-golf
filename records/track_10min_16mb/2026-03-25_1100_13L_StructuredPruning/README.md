# Exp 87: Progressive Pruning with Block Influence (13L → 11L)

## 목적
13L 모델로 학습 중 60% 시점에서 Block Influence (BI) 메트릭으로 가장 영향력이 낮은 2개 레이어를 제거하고, 나머지 40% 시간 동안 recovery fine-tuning하여 11L로 배포.
ShortGPT 논문의 in-place layer deletion + Model Healing 전략 적용.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)
- 11L, 512dim, 8 heads, 4 KV heads, MLP 3x, EMA(0.997), int6+zstd

## 변경 사항 (v2: Progressive Pruning)

| 파라미터 | Exp 73 | Exp 87 v2 | 근거 |
|----------|--------|-----------|------|
| NUM_LAYERS | 11 | **13** | 학습 시 더 많은 레이어로 표현력 확보 |
| PRUNE_LAYERS | - | **2** | BI 기반으로 2개 레이어 제거 |
| PRUNE_ENABLED | - | **1** | 프루닝 활성화 |
| PRUNE_AT_FRAC | - | **0.60** | 학습 60% 시점에서 프루닝 |
| VE_LAYERS | 9,10 | **11,12** | 13L 모델의 마지막 2개 레이어 |

## v1 실패 분석 (post-training pruning)
- v1은 학습 완료 후 pruning → recovery training 없이 배포
- 결과: post_ema 1.1290 → int6 roundtrip **2.3608** (catastrophic)
- 원인: skip_weights가 uniform(1.0)으로 재초기화, pruning 후 모델이 적응할 기회 없음
- 13L 전체 파라미터가 저장되어 크기 이점도 없음

## v2: Progressive Pruning 전략 (ShortGPT 기반)
1. **학습 60%** (wallclock 360초): 13L 전체 모델로 학습
2. **BI 측정**: validation 데이터로 각 레이어의 Block Influence 측정
3. **In-place 레이어 삭제**: `del model.blocks[idx]` (ShortGPT 스타일, 높은 인덱스부터)
4. **Model Healing**: 나머지 40% (240초) 동안 11L 모델로 recovery fine-tuning
5. **DDP 재구성**: 프루닝 후 torch.compile + DDP 재래핑
6. **옵티마이저 재구축**: 프루닝된 모델의 파라미터로 새 옵티마이저 그룹 생성
7. **EMA/SWA 리셋**: 프루닝된 모델 상태로 EMA 재초기화, SWA 카운트 리셋

## Block Influence (BI) 메트릭
- BI = 1 - cosine_similarity(layer_input, layer_output)
- BI가 높을수록 해당 레이어가 representation을 더 많이 변화시킴 (= 더 중요)
- BI가 낮은 레이어 = 거의 identity mapping → 제거 가능
- 첫 번째/마지막 레이어는 U-Net 앵커이므로 제거 불가 (후보: 인덱스 1~11)

## 프루닝 후 자동 업데이트
- U-Net encoder/decoder 분할 재계산
- skip_weights 재초기화 (recovery training으로 학습됨)
- XSA 플래그: 마지막 N개 레이어에 재할당
- VE 레이어: 마지막 2개 레이어로 재매핑

## 위험도
- **중~낮**: v1 대비 대폭 개선. 40% recovery training으로 skip_weights 등 재학습 가능
- torch.compile 재컴파일에 약간의 시간 소요 예상

## 예상 결과
- 13L 학습 → 11L pruning + recovery → post-training pruning 대비 큰 폭 개선
- val_bpb: ~1.1180~1.1220 (Exp 73 대비 소폭 개선 또는 동등)
- 크기: ~15.6MB (11L 모델, 프루닝된 파라미터만 저장)

## 환경 변수
- `NUM_LAYERS`: 학습 레이어 수 (기본값: 13)
- `PRUNE_LAYERS`: 제거할 레이어 수 (기본값: 2)
- `PRUNE_ENABLED`: 프루닝 활성화 여부 (기본값: 1)
- `PRUNE_AT_FRAC`: 프루닝 시점 (기본값: 0.60, wallclock 비율)
- `VE_LAYERS`: Value Embedding 적용 레이어 (기본값: "11,12")

## v1 실제 결과
- model_params: 32,241,260 (13L)
- post_ema val_bpb: 1.1290 (학습은 양호)
- BI scores: layer 9(0.087), layer 10(0.075) → 가장 낮아 pruning 대상
- pruned_layers: [9, 10], remaining: 11
- int6 roundtrip val_bpb: **2.3543** (catastrophic, recovery training 없음)
- 원인: post-training pruning으로 model healing 없음

## v2 실제 결과
(학습 후 기록 예정)
