# 10L SwiGLU + QAT

## 목적

현재 SOTA(10L, relu^2 MLP, val_bpb 1.14239)의 MLP를 SwiGLU로 교체하여 **동일 파라미터 수에서 MLP 표현력을 향상**시킨다. SwiGLU는 LLaMA, Gemma 등에서 검증된 활성화 함수로, 게이팅 메커니즘이 정보 흐름을 더 정밀하게 제어한다.

## 구현 방향

- **베이스**: 현재 SOTA (10L QAT, val_bpb 1.14239, 15.77MB)
- **핵심 변경**: MLP만 교체 (relu^2 → SwiGLU)
  - 기존: `fc(512→1536)` → `relu(x)^2` → `proj(1536→512)` (2 projections)
  - SwiGLU: `silu(gate(512→H)) * up(512→H)` → `down(H→512)` (3 projections)
  - 파라미터 수 동일하게 유지: H = dim × mlp_mult × 2/3 = 512 × 3.0 × 2/3 = 1024
  - 64의 배수로 정렬: H = 1024
- **양자화**: `_classify_param`이 `.mlp.`로 분류하므로 gate/up/down 모두 int5 QAT 자동 적용
- **초기화**: `down` projection에 zero_init + output scaling 적용 (기존 proj와 동일 역할)
- **그 외 설정**: 10L, int5/int6 QAT, BigramHash 10240, pruning 5%, cosine warmdown, SWA 0.35

## SwiGLU vs relu^2

```
relu^2:  y = proj(relu(fc(x))^2)         — 단순한 비선형 변환
SwiGLU:  y = down(silu(gate(x)) * up(x))  — 게이트가 정보 흐름 제어
```

- **SwiGLU 장점**: 게이팅이 입력에 따라 어떤 특징을 활성화할지 학습 → 더 정밀한 표현
- **SwiGLU 단점**: 3 projections → 같은 파라미터에서 hidden dim이 2/3로 축소 (1536→1024)
- **트레이드오프**: hidden 축소 vs 게이팅 품질 향상 — 대부분의 LLM에서 SwiGLU가 우세

## 예상 결과

- **val_bpb**: ~1.139~1.143 (SOTA와 비슷하거나 약간 개선)
- **아티팩트 크기**: ~15.8MB (파라미터 수 동일, 크기 거의 변화 없음)
- **학습 시간**: step_avg ~93ms (구조 변경 미미)
- **위험**: 중
  - relu^2의 희소성이 int5 양자화에 유리할 수 있음 (SwiGLU는 dense)
  - hidden dim 축소(1536→1024)가 게이팅 이점을 상쇄할 수 있음

## 실제 결과

(학습 후 기록 예정)
