# Q5. 자동차 부품 결함 검출 (딥러닝 기초 + 전이학습)

| 난이도 | 권장 시간 | 배점 |
|--------|----------|------|
| ★★★ | 40분 | 100점 |

---

## 문제

자동차 부품 이미지(`data/part_images/`)와 검사 기록(`data/inspection_log.csv`)을 이용하여
결함 유형을 **3가지 방식**으로 자동 분류하고 비교하세요.

> **주의:** 레이블 불균형(양품 다수) + 일부 손상 이미지 포함

`q5_detection.py`의 클래스/함수와 `main()`을 완성하여 `result_q5.json`을 생성하세요.

---

## 요구사항

### Part A: 데이터 전처리 (클래스 구현)

1. **`DefectImageLoader`**: 이미지 로드 → RGB 변환 → 64×64 리사이즈 → [0,1] 정규화 → flatten (12288차원). 손상 이미지는 건너뛰기.
2. **`InspectionLogProcessor`**: CSV 로드 → 중복 part_id 제거 → defect_type 정리 → 유효 레이블만 필터링 → NaN을 빈 문자열로 대체
   - 유효 레이블: `["양품", "스크래치", "크랙", "변색", "이물질"]`

### Part B: 3단계 모델 비교

train(70%) / test(30%), `random_state=42`, `stratify=labels`

**B-1. 규칙 기반 (엣지 강도)**
- `conv2d(image, kernel)` NumPy 직접 구현 + Sobel 커널
- 엣지 평균 강도 → 임계값(train 양품 중앙값) 기반 이진 분류

**B-2. ML 기반 (특징 추출 + LR)**
- 이미지: PCA (`n_components=0.95`)
- 텍스트: TfidfVectorizer (`max_features=100`)
- `np.hstack` 결합 → `LogisticRegression(C=1.0, max_iter=1000, random_state=42)`

**B-3. 2층 신경망 Forward Pass**
- `data/pretrained_nn_weights.npz`에서 가중치, `data/pretrained_features.npy`에서 특징 로드
- ReLU + Softmax 활성화 함수 NumPy 직접 구현
- 수치 안정성 주의

### Part C: 전이학습 비교

- **Scratch**: PCA 이미지 특징 → LR
- **Pretrained**: pretrained_features → LR
- `transfer_gain` = pretrained accuracy - scratch accuracy

### Part D: 성능 평가 + 개선 + 보고서

- Pretrained 모델: 클래스별 F1 (5개), Confusion Matrix (5×5)
- **개선**: 소수 클래스 성능 향상 기법 적용 → before/after F1 비교
- **보고서** 4개 섹션: purpose, key_results, transfer_learning_effect, improvement_suggestion

---

## 제약 사항
- `conv2d`: NumPy만 (`cv2.filter2D` 금지)
- NN Forward Pass: NumPy만 (`PyTorch`/`Keras` 금지)
- `PCA`, `TfidfVectorizer`, `LogisticRegression`은 sklearn 허용
- 수치 소수점 4자리 반올림

---

## 입력 파일

| 파일 | 설명 |
|------|------|
| `data/part_images/` | `0000.png` ~ `0499.png` (일부 손상) |
| `data/inspection_log.csv` | part_id, defect_type, inspector_note |
| `data/pretrained_features.npy` | 사전학습 특징 (500×128) |
| `data/pretrained_nn_weights.npz` | W1, b1, W2, b2, feature_mean, feature_std |

---

## 출력 (`result_q5.json`)

```json
{
  "data_summary": {
    "total_valid_samples": 정수,
    "label_distribution": {"양품": 정수, ...},
    "imbalance_ratio": 실수
  },
  "rule_based": {"test_accuracy": 실수, "method": "edge_threshold_binary"},
  "ml_based": {"test_accuracy": 실수, "test_f1_macro": 실수, "pca_n_components": 정수},
  "nn_forward": {"test_accuracy": 실수, "test_f1_macro": 실수},
  "pretrained": {
    "test_accuracy": 실수, "test_f1_macro": 실수,
    "class_f1": {"양품": 실수, ...},
    "confusion_matrix": [[정수 5개], ...]
  },
  "transfer_gain": 실수,
  "improvement": {"before_f1": 실수, "after_f1": 실수, "most_improved_class": "문자열"},
  "report": {"purpose": "...", "key_results": "...", "transfer_learning_effect": "...", "improvement_suggestion": "..."}
}
```

---

## 채점 기준 (10항목, 각 10점)

| 항목 | 검증 내용 |
|------|----------|
| 클래스 구현 | DefectImageLoader + InspectionLogProcessor 존재 |
| 데이터 요약 | total_valid_samples ±2, label_distribution ±2 |
| conv2d + Sobel | conv2d 함수 + filter2D 미사용 + Sobel 커널 |
| 규칙 기반 | accuracy 0.65~0.85, method 일치 |
| ML 기반 | accuracy > 0.93, f1 > 0.85, PCA 100~400 |
| NN Forward | accuracy > 0.9 |
| 전이학습 | pretrained acc > 0.93, class_f1 5개, CM 5×5 |
| 개선 실험 | before_f1 > 0, after_f1 > 0, class in LABELS |
| 보고서 | 4개 섹션, 각 ≥ 10자 |
| 라이브러리 | TfidfVectorizer+LR+PCA 사용, keras/torch 미사용 |
