# test_q1_svd.py  —  Q1 SVD (5 tests, 100점)
import json
import os
import pathlib

import pytest

import q1_svd as stu

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
DATA_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "questions" / "q1_svd" / "data"
)
QUESTION_DIR = DATA_DIR.parent

# ---------------------------------------------------------------------------
# 기대값 (참조 솔루션 실행 결과)
# ---------------------------------------------------------------------------
EXPECTED_OPTIMAL_K = 35
EXPECTED_CUMVAR = 0.950775
EXPECTED_MSE = 0.04824
EXPECTED_TOP5_SV = [99.673366, 83.66078, 78.648015, 73.235932, 62.823237]
EXPECTED_EVR5 = [0.202751, 0.142839, 0.126235, 0.109459, 0.080546]

# ---------------------------------------------------------------------------
# Fixture: main() 1회 실행 → result JSON 캐싱
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def result():
    orig = os.getcwd()
    os.chdir(QUESTION_DIR)
    try:
        stu.main()
        with open("result_q1.json", "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        os.chdir(orig)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_optimal_k(result):
    """[20점] optimal_k 정확 일치"""
    assert result.get("optimal_k") == EXPECTED_OPTIMAL_K


def test_cumulative_variance(result):
    """[20점] cumulative_variance_at_k (허용 오차 < 1e-4)"""
    val = result.get("cumulative_variance_at_k")
    assert val is not None, "cumulative_variance_at_k 누락"
    assert abs(val - EXPECTED_CUMVAR) < 1e-4, (
        f"결과: {val}, 기대: {EXPECTED_CUMVAR}"
    )


def test_reconstruction_mse(result):
    """[20점] reconstruction_mse (허용 오차 < 1e-4)"""
    val = result.get("reconstruction_mse")
    assert val is not None, "reconstruction_mse 누락"
    assert abs(val - EXPECTED_MSE) < 1e-4, (
        f"결과: {val}, 기대: {EXPECTED_MSE}"
    )


def test_top5_singular_values(result):
    """[20점] top_5_singular_values (각 허용 오차 < 1e-3)"""
    sv = result.get("top_5_singular_values", [])
    assert len(sv) == 5, f"5개 필요, {len(sv)}개 제출"
    for i, (a, b) in enumerate(zip(sv, EXPECTED_TOP5_SV)):
        assert abs(a - b) < 1e-3, f"S[{i}]: 결과 {a}, 기대 {b}"


def test_explained_variance_ratio(result):
    """[20점] explained_variance_ratio_top5 (각 허용 오차 < 1e-4)"""
    evr = result.get("explained_variance_ratio_top5", [])
    assert len(evr) == 5, f"5개 필요, {len(evr)}개 제출"
    for i, (a, b) in enumerate(zip(evr, EXPECTED_EVR5)):
        assert abs(a - b) < 1e-4, f"EVR[{i}]: 결과 {a}, 기대 {b}"
