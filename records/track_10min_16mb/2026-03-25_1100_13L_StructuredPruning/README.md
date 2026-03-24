# Exp 87: Structured Pruning with Block Influence (13L → 11L)

## 목적
13L 모델로 학습하여 더 나은 표현력을 확보한 후, Block Influence (BI) 메트릭으로 가장 영향력이 낮은 2개 레이어를 제거하여 11L로 배포.
더 많은 레이어로 학습하면 더 좋은 representation을 얻고, 중요도가 낮은 레이어를 제거해도 성능 손실이 최소화된다는 가설.

## 베이스
`2026-03-24_2100_11L_LeakyReLU_Bigram6144/train_gpt.py` (Exp 73, val_bpb 1.1221)
- 11L, 512dim, 8 heads, 4 KV heads, MLP 3x, EMA(0.997), int6+zstd

## 변경 사항

| 파라미터 | Exp 73 | Exp 87 | 근거 |
|----------|--------|--------|------|
| NUM_LAYERS | 11 | **13** | 학습 시 더 많은 레이어로 표현력 확보 |
| PRUNE_LAYERS | - | **2** | 학습 후 BI 기반으로 2개 레이어 제거 |
| PRUNE_ENABLED | - | **1** | 프루닝 활성화 |
| VE_LAYERS | 9,10 | **11,12** | 13L 모델의 마지막 2개 레이어 |

## Block Influence (BI) 메트릭
- BI = 1 - cosine_similarity(layer_input, layer_output)
- 각 레이어의 입력과 출력 간 코사인 유사도를 측정
- BI가 높을수록 해당 레이어가 representation을 더 많이 변화시킴 (= 더 중요)
- BI가 낮은 레이어는 입력을 거의 그대로 통과시키므로 제거 가능

## 프루닝 전략
- 학습 완료 후, validation 데이터로 각 레이어의 BI 측정
- 첫 번째/마지막 레이어는 U-Net 구조의 앵커이므로 제거 불가
- 중간 레이어 (인덱스 1~11) 중 BI가 가장 낮은 2개 제거
- 제거 후 레이어 인덱스 재배치, skip_weights 재계산
- VE 레이어 인덱스도 재매핑

## U-Net 재인덱싱
- 프루닝 후 남은 레이어를 0부터 재인덱싱
- skip_weights는 새로운 encoder/decoder 분할에 맞게 uniform(1.0)으로 재초기화
- VE 레이어 인덱스는 생존한 레이어의 새 인덱스로 매핑

## 위험도
- **중간**: 프루닝 후 fine-tuning 없이 바로 배포 (recovery training 미적용)
- skip_weights가 uniform으로 재초기화되므로 약간의 성능 저하 가능

## 예상 결과
- 13L 학습으로 더 나은 representation → 프루닝 후에도 11L 직접 학습보다 우수
- val_bpb: ~1.1180~1.1220 (Exp 73 대비 소폭 개선 또는 동등)
- 크기: ~15.9MB (11L 모델과 동일)

## 환경 변수
- `NUM_LAYERS`: 학습 레이어 수 (기본값: 13)
- `PRUNE_LAYERS`: 제거할 레이어 수 (기본값: 2)
- `PRUNE_ENABLED`: 프루닝 활성화 여부 (기본값: 1)
- `VE_LAYERS`: Value Embedding 적용 레이어 (기본값: "11,12")

## 실제 결과
(학습 후 기록 예정)
