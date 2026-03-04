# test_q3_cv.py  —  Q3 CV (7 tests, 100점)
import ast
import json
import os
import pathlib
import re

import pytest

import q3_cv as stu

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
QUESTION_DIR = pathlib.Path(__file__).resolve().parent
CODE_PATH = pathlib.Path(__file__).resolve().parent / "q3_cv.py"

# ---------------------------------------------------------------------------
# 소규모 데이터 (embed)
# ---------------------------------------------------------------------------
LABELS = {
    "easy_01": 3, "easy_02": 4, "easy_03": 5, "easy_04": 3, "easy_05": 6,
    "medium_01": 4, "medium_02": 5, "medium_03": 6, "medium_04": 7, "medium_05": 5,
    "hard_01": 5, "hard_02": 7, "hard_03": 8, "hard_04": 6, "hard_05": 9,
    "test_01": 10,
}

VALID_IMAGES = {
    "easy_01", "easy_02", "easy_03", "easy_04", "easy_05",
    "medium_01", "medium_02", "medium_03", "medium_04", "medium_05",
    "hard_01", "hard_02", "hard_03", "hard_04", "hard_05",
}

KR_PATTERN = re.compile(r"[가-힣]")

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
def test_conv2d_impl(student_code):
    """[15점] conv2d 함수 존재 + filter2D 미사용"""
    tree = ast.parse(student_code)
    funcs = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    assert "conv2d" in funcs, "conv2d 함수 미구현"
    assert "filter2D" not in student_code, "filter2D 사용 감지 — 직접 구현 필요"


def test_valid_images(result):
    """[15점] predictions 키 = 15개 유효 이미지 (test_01 제외)"""
    preds = result.get("predictions", {})
    assert set(preds.keys()) == VALID_IMAGES, (
        f"extra={set(preds.keys()) - VALID_IMAGES}, "
        f"missing={VALID_IMAGES - set(preds.keys())}"
    )


def test_easy_mae(result):
    """[15점] easy 카테고리 MAE (0→0점, ≤3→만점)"""
    preds = result.get("predictions", {})
    easy_keys = [k for k in LABELS if k.startswith("easy") and k in VALID_IMAGES]
    errors = [abs(preds.get(k, 0) - LABELS[k]) for k in easy_keys]
    mae = sum(errors) / len(errors) if errors else 999
    assert mae != 0.0, "MAE=0.0: 과적합 또는 데이터 누수 의심"
    assert mae <= 3.0, f"easy MAE={mae:.2f} (3.0 이하 필요)"


def test_metrics_categories(result):
    """[10점] metrics에 easy/medium/hard 3카테고리 포함"""
    m = result.get("metrics", {})
    for cat in ["easy", "medium", "hard"]:
        assert cat in m, f"metrics에 '{cat}' 카테고리 누락"


def test_failure_reasons(result):
    """[20점] failure_reasons: 3개 이상, 각 20자+, 한국어 포함"""
    fr = result.get("failure_reasons", [])
    assert len(fr) >= 3, f"{len(fr)}개 제출 (최소 3개 필요)"
    for i, reason in enumerate(fr):
        assert len(reason) >= 20, f"failure_reasons[{i}]: {len(reason)}자 (20자 이상 필요)"
        assert KR_PATTERN.search(reason), f"failure_reasons[{i}]: 한국어 미포함"


def test_why_learning_based(result):
    """[15점] why_learning_based: 30~200자, 한국어 포함"""
    wlb = result.get("why_learning_based", "")
    assert 30 <= len(wlb) <= 200, f"길이: {len(wlb)}자 (30~200자 필요)"
    assert KR_PATTERN.search(wlb), "한국어 미포함"


def test_worst_case(result):
    """[10점] worst_case_image가 hard 카테고리"""
    wc = result.get("worst_case_image", "")
    assert wc.startswith("hard"), f"결과: '{wc}' ('hard'로 시작해야 함)"
