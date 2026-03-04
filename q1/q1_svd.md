# Q1. SVD 기반 데이터 차원 축소 및 복원 분석

| 난이도 | 권장 시간 | 배점 |
|--------|----------|------|
| ★☆☆ | 15분 | 100점 |

---

## 문제

공장 100개 센서에서 수집된 500건 측정 데이터(`data/sensor_data.csv`, 500×100)가 주어집니다.
SVD를 이용해 핵심 패턴만 보존하면서 노이즈를 제거하세요.

`q1_svd.py`의 `main()` 함수를 완성하여 `result_q1.json`을 생성하세요.

---

## 요구사항

### 1. 데이터 로드 및 전처리
- `sensor_data.csv` 로드 (헤더 없음)
- 각 열을 **평균 0, 표준편차 1**로 표준화 (NumPy만 사용)
- 표준편차: `ddof=0`

### 2. SVD 분해
- `numpy.linalg.svd(X, full_matrices=False)`로 U, S, Vt 분해

### 3. Explained Variance Ratio
```
explained_variance_ratio[i] = S[i]² / sum(S²)
```

### 4. 최적 k 결정
- Cumulative Variance Ratio가 **처음으로 95% 이상**이 되는 최소 k

### 5. 차원 축소 및 복원
```
X_reduced = U[:, :k] * S[:k]
X_reconstructed = X_reduced @ Vt[:k, :]
```

### 6. 복원 오차
```
MSE = mean((X - X_reconstructed)²)
```

---

## 제약 사항
- **NumPy만 사용** (`sklearn`/`scipy` 금지, `pandas`는 로드에만 허용)
- `full_matrices=False` 필수
- 소수점 6자리 반올림

---

## 출력 (`result_q1.json`)

```json
{
  "optimal_k": 정수,
  "cumulative_variance_at_k": 실수,
  "reconstruction_mse": 실수,
  "top_5_singular_values": [실수 5개],
  "explained_variance_ratio_top5": [실수 5개]
}
```

---

## 채점 기준 (5항목, 각 20점)

| 항목 | 검증 내용 |
|------|----------|
| optimal_k | 정확 일치 |
| cumulative_variance_at_k | 오차 < 0.0001 |
| reconstruction_mse | 오차 < 0.0001 |
| top_5_singular_values | 각 오차 < 0.001 |
| explained_variance_ratio_top5 | 각 오차 < 0.0001 |
