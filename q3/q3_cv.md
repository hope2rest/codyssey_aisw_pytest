# Q3. 이미지 기반 객체 카운팅 (규칙 기반 구현 및 한계 분석)

| 난이도 | 권장 시간 | 배점 |
|--------|----------|------|
| ★★☆ | 25분 | 100점 |

---

## 문제

물류 창고 컨베이어 벨트 위 박스 개수를 자동으로 세는 시스템을 만드세요.

이미지는 3개 카테고리로 구분됩니다:

| 카테고리 | 장수 | 특징 |
|----------|------|------|
| `easy` | 5장 | 밝은 배경, 박스 간 간격 충분 |
| `medium` | 5장 | 일부 겹침, 약간의 그림자 |
| `hard` | 5장 | 적재 형태, 밀집/겹침 |

`q3_cv.py`의 함수들과 `main()`을 완성하여 `result_q3.json`을 생성하세요.

---

## 요구사항

### Part A: 엣지 검출 (직접 구현)

1. **`conv2d(image, kernel)`**: NumPy만으로 2D 컨볼루션 (valid 모드)
2. **Sobel 커널(3×3)** 으로 수평/수직 엣지 검출:
   ```
   edge_magnitude = sqrt(Gx² + Gy²)
   ```

### Part B: 박스 카운팅

3. 엣지 이미지 **이진화(thresholding)**
4. **Connected Component 분석**으로 박스 개수 추정
   - 직접 구현(BFS/DFS) 또는 `scipy.ndimage.label` 사용 가능
5. **최소 면적 필터**로 노이즈 제거

### Part C: 성능 분석

6. 카테고리별 **MAE**, **Accuracy** 계산
7. hard에서 가장 큰 오차 이미지에 대해 **실패 원인 3가지 이상** 서술 (각 20자 이상)
8. **학습 기반 접근법이 필요한 이유** 200자 이내 서술

---

## 제약 사항
- `conv2d`는 **NumPy만으로 직접 구현** (`cv2.filter2D` 사용 금지)
- 이미지 로드: `PIL` 또는 `cv2` 사용 가능
- 그레이스케일: `gray = 0.299*R + 0.587*G + 0.114*B`

---

## 입력 파일

| 파일 | 설명 |
|------|------|
| `data/images/` | `easy_01.png` ~ `hard_05.png` (640×480 RGB) |
| `data/labels.json` | `{"easy_01": 3, "easy_02": 5, ...}` |

---

## 출력 (`result_q3.json`)

```json
{
  "predictions": {"easy_01": 정수, ...},
  "metrics": {
    "easy":   {"mae": 실수, "accuracy": 실수},
    "medium": {"mae": 실수, "accuracy": 실수},
    "hard":   {"mae": 실수, "accuracy": 실수}
  },
  "worst_case_image": "이미지명",
  "failure_reasons": ["이유1", "이유2", "이유3"],
  "why_learning_based": "서술"
}
```

---

## 채점 기준 (7항목, 총 100점)

| 항목 | 배점 | 검증 내용 |
|------|------|----------|
| conv2d 구현 | 15 | 함수 존재 + filter2D 미사용 |
| 유효 이미지 | 15 | predictions 키 = 15개 |
| easy MAE | 15 | MAE ≤ 3 |
| metrics 구조 | 10 | easy/medium/hard + mae/accuracy |
| 실패 원인 | 20 | ≥ 3개, 각 ≥ 20자, 한국어 |
| 학습 기반 이유 | 15 | 30~200자, 한국어 |
| worst_case | 10 | "hard"로 시작 |
