# Q2. TF-IDF 직접 구현 및 코사인 유사도 문서 검색

| 난이도 | 권장 시간 | 배점 |
|--------|----------|------|
| ★☆☆ | 15분 | 100점 |

---

## 문제

20개 한국어 기술 문서(`data/documents.txt`)가 주어집니다.
TF-IDF와 코사인 유사도를 직접 구현하여 검색 쿼리에 가장 유사한 문서 상위 3개를 반환하세요.

`q2_tfidf.py`의 함수들과 `main()`을 완성하여 `result_q2.json`을 생성하세요.

---

## 요구사항

### 1. 전처리 함수 `preprocess(text, stopwords)`
- 소문자 변환
- 정규표현식으로 **한글, 영문, 숫자 외 문자 제거**
- 공백 기준 토큰화
- 불용어(`stopwords.txt`) 제거
- 길이 1 이하 토큰 제거

### 2. TF 계산
```
TF(t, d) = count(t in d) / total_words(d)
```

### 3. IDF 계산 (Smooth IDF)
```
IDF(t) = log((N + 1) / (df(t) + 1)) + 1
```

### 4. TF-IDF 행렬
- (문서 수 × 단어 수) NumPy 배열
- 단어(열)는 **사전순 정렬**

### 5. 코사인 유사도 `cosine_similarity(a, b)`
```
cosine_sim(a, b) = dot(a, b) / (norm(a) * norm(b))
```
- 영벡터(norm=0)이면 0.0 반환

### 6. 검색 함수 `search()`
- 쿼리에 동일한 전처리 + TF-IDF 적용
- 각 문서와 코사인 유사도 계산 → 상위 3개 반환

---

## 제약 사항
- **NumPy만 사용** (`sklearn`/`scipy` 금지)
- Smooth IDF 수식 정확히 적용
- 쿼리 IDF는 기존 코퍼스 기준 (쿼리를 코퍼스에 추가하지 않음)
- 유사도 소수점 6자리 반올림

---

## 입력 파일

| 파일 | 설명 |
|------|------|
| `data/documents.txt` | 한 줄에 하나의 문서 (20줄) |
| `data/stopwords.txt` | 한 줄에 하나의 불용어 |
| `data/queries.txt` | 검색 쿼리 (5개) |

---

## 출력 (`result_q2.json`)

```json
{
  "num_documents": 정수,
  "vocab_size": 정수,
  "tfidf_matrix_shape": [정수, 정수],
  "search_results": [
    {
      "query": "쿼리 텍스트",
      "top3": [
        {"doc_index": 정수, "similarity": 실수},
        ...
      ]
    },
    ...
  ]
}
```

---

## 채점 기준 (8항목, 총 100점)

| 항목 | 배점 | 검증 내용 |
|------|------|----------|
| num_documents | 10 | == 20 |
| vocab_size | 20 | 정확 일치 |
| tfidf_shape | 5 | 정확 일치 |
| 쿼리 1~4 | 각 13 | doc_index 일치 + similarity 오차 < 0.001 |
| 쿼리 5 | 13 | 불용어만 쿼리 → 유사도 모두 0.0 |
