# test_q4_sentiment.py  —  Q4 Sentiment (7 tests, 100점)
import json
import os
import pathlib

import pytest

import q4_sentiment as stu

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
QUESTION_DIR = pathlib.Path(__file__).resolve().parent
CODE_PATH = pathlib.Path(__file__).resolve().parent / "q4_sentiment.py"

# ---------------------------------------------------------------------------
# 소규모 데이터 (embed) — 감성 사전
# ---------------------------------------------------------------------------
SENTIMENT_DICT = {
    "positive": {
        "좋다": 1.0, "좋아요": 1.0, "좋은": 1.0, "좋습니다": 1.0,
        "훌륭하다": 1.5, "훌륭한": 1.5, "훌륭합니다": 1.5,
        "만족": 1.0, "만족합니다": 1.0, "만족스럽다": 1.2,
        "최고": 1.5, "최고입니다": 1.5, "최고예요": 1.5,
        "추천": 1.0, "추천합니다": 1.0,
        "편리하다": 1.0, "편리합니다": 1.0,
        "빠르다": 0.8, "빠른": 0.8, "빠릅니다": 0.8,
        "깔끔하다": 1.0, "깔끔한": 1.0,
        "예쁘다": 1.0, "예쁜": 1.0,
        "우수한": 1.3, "뛰어난": 1.3,
        "완벽한": 1.5, "완벽합니다": 1.5,
        "괜찮다": 0.5, "맛있다": 1.0, "신선한": 1.0,
        "저렴하다": 0.8, "가성비": 1.0,
    },
    "negative": {
        "나쁘다": -1.0, "나쁜": -1.0, "좋다": -0.3,
        "불만": -1.5, "불만족": -1.5, "불만입니다": -1.5,
        "실망": -1.5, "실망입니다": -1.5, "실망스럽다": -1.5,
        "별로": -1.0, "별로입니다": -1.0,
        "느리다": -0.8, "느린": -0.8,
        "비싸다": -0.8, "비싼": -0.8,
        "불편하다": -1.0, "불편한": -1.0,
        "고장": -1.5, "고장났다": -1.5,
        "후회": -1.5, "후회합니다": -1.5,
        "최악": -2.0, "최악입니다": -2.0,
        "불량": -1.5, "부족하다": -0.8,
        "파손": -1.5, "냄새": -0.8,
    },
    "negation": ["안", "않", "못", "없"],
    "intensifier": {
        "매우": 1.5, "정말": 1.3, "너무": 1.2, "아주": 1.3,
        "진짜": 1.3, "상당히": 1.2, "굉장히": 1.4, "엄청": 1.3,
    },
}

# ---------------------------------------------------------------------------
# 기대값
# ---------------------------------------------------------------------------
EXPECTED_RULE_ACC = 0.9524
EXPECTED_ML_ACC = 0.9947
EXPECTED_ML_F1 = 0.9893

REQUIRED_METRICS = [
    "accuracy", "precision_pos", "recall_pos",
    "precision_neg", "recall_neg", "f1_macro",
]

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result():
    orig = os.getcwd()
    os.chdir(QUESTION_DIR)
    try:
        return stu.main()
    finally:
        os.chdir(orig)


@pytest.fixture(scope="module")
def student_code():
    with open(CODE_PATH, "r", encoding="utf-8") as f:
        return f.read()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_rule_based_metrics(result):
    """[10점] rule_based 6개 지표 + accuracy 검증 (오차 < 0.03)"""
    rb = result.get("rule_based", {})
    missing = [k for k in REQUIRED_METRICS if k not in rb]
    assert not missing, f"누락 지표: {missing}"
    rb_acc = rb.get("accuracy", 0)
    assert abs(rb_acc - EXPECTED_RULE_ACC) <= 0.03, (
        f"accuracy 불일치: 결과 {rb_acc}, 기대 {EXPECTED_RULE_ACC}"
    )


def test_ml_accuracy(result):
    """[15점] ml_based accuracy (오차 < 0.01)"""
    ml_acc = result.get("ml_based", {}).get("accuracy", 0)
    assert abs(ml_acc - EXPECTED_ML_ACC) < 0.01, (
        f"결과: {ml_acc}, 기대: {EXPECTED_ML_ACC}"
    )


def test_ml_f1(result):
    """[15점] ml_based f1_macro (오차 < 0.01)"""
    ml_f1 = result.get("ml_based", {}).get("f1_macro", 0)
    assert abs(ml_f1 - EXPECTED_ML_F1) < 0.01, (
        f"결과: {ml_f1}, 기대: {EXPECTED_ML_F1}"
    )


def test_shap_signs(result):
    """[15점] SHAP positive 5개 > 0, negative 5개 < 0"""
    sp = result.get("shap_top5_positive", [])
    sn = result.get("shap_top5_negative", [])
    assert len(sp) == 5, f"positive: {len(sp)}개 (5개 필요)"
    assert len(sn) == 5, f"negative: {len(sn)}개 (5개 필요)"
    assert all(x.get("shap_value", 0) > 0 for x in sp), "positive 항목 중 shap_value <= 0 존재"
    assert all(x.get("shap_value", 0) < 0 for x in sn), "negative 항목 중 shap_value >= 0 존재"


def test_fit_transform(student_code):
    """[15점] fit_transform + .transform( 분리 확인 (데이터 누수 방지)"""
    assert "fit_transform" in student_code, "fit_transform 미사용"
    assert ".transform(" in student_code, ".transform() 미사용 — fit/transform 분리 필요"


def test_business_summary(result):
    """[15점] business_summary: '긍정'+'부정' 포함, 20자 이상"""
    bs = result.get("business_summary", "")
    assert "긍정" in bs, "'긍정' 키워드 누락"
    assert "부정" in bs, "'부정' 키워드 누락"
    assert len(bs) >= 20, f"길이 부족: {len(bs)}자 (20자 이상 필요)"


def test_ml_metrics_complete(result):
    """[15점] ml_based에 6개 지표 키 모두 존재"""
    ml = result.get("ml_based", {})
    missing = [k for k in REQUIRED_METRICS if k not in ml]
    assert not missing, f"누락 지표: {missing}"
