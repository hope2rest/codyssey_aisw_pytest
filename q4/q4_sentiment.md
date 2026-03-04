# Q4. 고객 리뷰 감성 분석 (규칙 기반 + ML + SHAP 해석)

| 난이도 | 권장 시간 | 배점 |
|--------|----------|------|
| ★★☆ | 25분 | 100점 |

---

## 문제

이커머스 고객 리뷰 데이터(`data/reviews.csv`)에 대해 감성 분석을 **두 가지 방식**으로 구축하고 비교하세요.

> **주의:** label 분포는 **불균형** 상태입니다.

`q4_sentiment.py`의 함수들과 `main()`을 완성하여 `result_q4.json`을 생성하세요.

---

## 요구사항

### Part A: 규칙 기반 감성 분석

1. **감성 사전**(`data/sentiment_dict.json`) 로드
2. **감성 점수 산출:**
   - 텍스트를 토큰 단위로 분리
   - 각 토큰의 감성 점수를 사전에서 조회
   - **부정어** 바로 다음 토큰: 감성 점수 × (-1)
   - **강조어** 바로 다음 토큰: 감성 점수 × 해당 배수
   - 총점 > 0 → 긍정(1), ≤ 0 → 부정(0)

### Part B: ML 기반 감성 분석

3. train(70%) / test(30%) 분할 (`random_state=42`)
4. train 데이터의 클래스 **불균형 해소**
5. `TfidfVectorizer`로 벡터화 — **fit은 train에서만**, test는 transform만
6. `LogisticRegression(random_state=42)` 학습

### Part C: 비교 분석 + SHAP

7. 두 접근법의 성능 비교: Accuracy, Precision/Recall (긍정·부정), F1 Macro
8. `shap.LinearExplainer`로 SHAP 값 계산
   - 긍정 기여 상위 5개 단어 + SHAP 값
   - 부정 기여 상위 5개 단어 + SHAP 값
9. **비즈니스 요약**: "긍정", "부정" 키워드 포함, 20자 이상

---

## 제약 사항
- 규칙 기반: `sklearn` 사용 금지
- TF-IDF fit/transform **분리 필수** (데이터 누수 방지)
- 불균형 처리는 **train에서만**
- 수치 소수점 4자리 반올림

---

## 입력 파일

| 파일 | 설명 |
|------|------|
| `data/reviews.csv` | `text`, `label` (0 또는 1) |
| `data/sentiment_dict.json` | positive, negative, negation, intensifier |

---

## 출력 (`result_q4.json`)

```json
{
  "rule_based": {
    "accuracy": 실수, "precision_pos": 실수, "recall_pos": 실수,
    "precision_neg": 실수, "recall_neg": 실수, "f1_macro": 실수
  },
  "ml_based": { ... 동일 6개 지표 ... },
  "shap_top5_positive": [{"word": "문자열", "shap_value": 실수}, ...],
  "shap_top5_negative": [{"word": "문자열", "shap_value": 실수}, ...],
  "business_summary": "비기술적 요약"
}
```

---

## 채점 기준 (7항목, 총 100점)

| 항목 | 배점 | 검증 내용 |
|------|------|----------|
| rule_based 지표 | 10 | 6개 지표 + accuracy 오차 ≤ 0.03 |
| ML accuracy | 15 | 오차 < 0.01 |
| ML f1_macro | 15 | 오차 < 0.01 |
| SHAP 부호 | 15 | positive 5개 > 0, negative 5개 < 0 |
| fit/transform 분리 | 15 | 코드에 fit_transform + .transform( 존재 |
| 비즈니스 요약 | 15 | "긍정"+"부정" 포함, ≥ 20자 |
| ML 지표 완성도 | 15 | 6개 키 모두 존재 |
