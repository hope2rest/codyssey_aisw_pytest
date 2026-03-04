# codyssey_aisw_pytest

AI/ML 5개 문항 pytest 기반 자동 채점 시스템

## 구조

```
q1/  SVD 기반 차원축소 복원분석      (5 tests,  100점)
q2/  TF-IDF 문서 검색               (8 tests,  100점)
q3/  컴퓨터 비전 객체 탐지           (7 tests,  100점)
q4/  감성 분석                      (7 tests,  100점)
q5/  부품 결함 검출                  (10 tests, 100점)
```

각 폴더에 3개 파일:

| 파일 | 용도 |
|------|------|
| `q{N}_{name}.py` | 학생 제출용 skeleton |
| `q{N}_{name}_solution.py` | 정답지 |
| `test_q{N}_{name}.py` | pytest 채점 테스트 |

## 실행 방법

```bash
# 정답지로 테스트 (예: Q1)
cd q1
cp q1_svd_solution.py q1_svd.py
pytest test_q1_svd.py -v
```

## 의존성

```
numpy, pandas, scikit-learn, scipy, Pillow, shap, pytest
```
