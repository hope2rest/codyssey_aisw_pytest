# test_q5_detection.py  —  Q5 Detection (10 tests, 100점)
import ast
import json
import os
import pathlib

import pytest

import q5_detection as stu

# ---------------------------------------------------------------------------
# 경로
# ---------------------------------------------------------------------------
DATA_DIR = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "questions" / "q5_detection" / "data"
)
QUESTION_DIR = DATA_DIR.parent
CODE_PATH = pathlib.Path(__file__).resolve().parent / "q5_detection.py"

# ---------------------------------------------------------------------------
# 기대값
# ---------------------------------------------------------------------------
LABELS = ["양품", "스크래치", "크랙", "변색", "이물질"]
EXPECTED_VALID_SAMPLES = 495
EXPECTED_LABEL_DIST = {"양품": 296, "스크래치": 75, "크랙": 50, "변색": 49, "이물질": 25}

REPORT_SECTIONS = [
    "purpose",
    "key_results",
    "transfer_learning_effect",
    "improvement_suggestion",
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
def test_classes_impl(student_code):
    """[10점] DefectImageLoader + InspectionLogProcessor 클래스 구현"""
    tree = ast.parse(student_code)
    cls_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    assert "DefectImageLoader" in cls_names, "DefectImageLoader 클래스 미구현"
    assert "InspectionLogProcessor" in cls_names, "InspectionLogProcessor 클래스 미구현"


def test_data_summary(result):
    """[10점] total_valid_samples ±2, label_distribution ±2"""
    ds = result.get("data_summary", {})
    tvs = ds.get("total_valid_samples", 0)
    assert abs(tvs - EXPECTED_VALID_SAMPLES) <= 2, (
        f"유효 샘플 수 불일치: 결과 {tvs}, 기대 {EXPECTED_VALID_SAMPLES}"
    )
    ld = ds.get("label_distribution", {})
    for lbl in LABELS:
        exp = EXPECTED_LABEL_DIST.get(lbl, 0)
        got = ld.get(lbl, 0)
        assert abs(exp - got) <= 2, f"레이블 분포 불일치: {lbl} (결과 {got}, 기대 {exp})"


def test_conv2d_sobel(student_code):
    """[10점] conv2d 직접 구현 + filter2D 미사용 + Sobel 커널"""
    assert "def conv2d" in student_code, "conv2d 함수 없음"
    assert "filter2D" not in student_code, "filter2D 사용 감지"
    has_sobel = (
        "sobel" in student_code.lower()
        or ("[-1, 0, 1]" in student_code and "[-2, 0, 2]" in student_code)
        or "[-1,0,1]" in student_code
    )
    assert has_sobel, "Sobel 커널 미정의"


def test_rule_based(result):
    """[10점] 규칙 기반 accuracy 0.65~0.85, method=edge_threshold_binary"""
    rb = result.get("rule_based", {})
    rb_acc = rb.get("test_accuracy", 0)
    assert 0.65 <= rb_acc <= 0.85, f"accuracy={rb_acc} (0.65~0.85 필요)"
    assert rb.get("method") == "edge_threshold_binary", (
        f"method={rb.get('method')} (edge_threshold_binary 필요)"
    )


def test_ml_based(result):
    """[10점] ML accuracy > 0.93, f1 > 0.85, PCA 100~400"""
    ml = result.get("ml_based", {})
    assert ml.get("test_accuracy", 0) > 0.93, f"accuracy={ml.get('test_accuracy')}"
    assert ml.get("test_f1_macro", 0) > 0.85, f"f1={ml.get('test_f1_macro')}"
    pca_n = ml.get("pca_n_components", 0)
    assert 100 <= pca_n <= 400, f"PCA n_components={pca_n} (100~400 필요)"


def test_nn_forward(result, student_code):
    """[10점] NN accuracy > 0.9, ReLU/Softmax 구현, 가중치 로드"""
    assert result.get("nn_forward", {}).get("test_accuracy", 0) > 0.9, "NN accuracy <= 0.9"
    has_relu = "relu" in student_code.lower() or "maximum(0" in student_code
    assert has_relu, "ReLU 미구현"
    has_softmax = "softmax" in student_code.lower() or "exp(" in student_code
    assert has_softmax, "Softmax 미구현"
    assert "pretrained_nn_weights" in student_code, "사전학습 가중치 미로드"


def test_transfer(result):
    """[10점] pretrained acc > 0.93, class_f1 5개, confusion_matrix 5x5"""
    pre = result.get("pretrained", {})
    assert pre.get("test_accuracy", 0) > 0.93, f"pretrained accuracy={pre.get('test_accuracy')}"
    tg = result.get("transfer_gain")
    assert tg is not None and isinstance(tg, (int, float)), f"transfer_gain={tg}"
    cf1 = pre.get("class_f1", {})
    assert all(lbl in cf1 for lbl in LABELS), f"class_f1 누락: {[l for l in LABELS if l not in cf1]}"
    cm = pre.get("confusion_matrix", [])
    assert len(cm) == 5 and all(len(row) == 5 for row in cm), "confusion_matrix가 5x5 아님"


def test_improvement(result, student_code):
    """[10점] before_f1 > 0, after_f1 > 0, class_weight='balanced' 사용"""
    imp = result.get("improvement", {})
    assert imp.get("before_f1", 0) > 0, "before_f1 <= 0"
    assert imp.get("after_f1", 0) > 0, "after_f1 <= 0"
    assert imp.get("most_improved_class", "") in LABELS, "most_improved_class 유효하지 않음"
    assert "class_weight" in student_code and "balanced" in student_code, (
        "class_weight='balanced' 미사용"
    )


def test_report(result):
    """[10점] 보고서 4개 섹션, 각 10자 이상"""
    report = result.get("report", {})
    for sec in REPORT_SECTIONS:
        val = report.get(sec, "")
        assert isinstance(val, str) and len(val) >= 10, (
            f"report['{sec}']: 내용 부족"
        )


def test_libraries(student_code):
    """[10점] TfidfVectorizer+LogisticRegression+PCA 사용, keras/torch 미사용"""
    assert "TfidfVectorizer" in student_code, "TfidfVectorizer 미사용"
    assert "LogisticRegression" in student_code, "LogisticRegression 미사용"
    assert "PCA" in student_code, "PCA 미사용"
    low = student_code.lower()
    assert "keras" not in low, "keras 사용 감지"
    assert "torch" not in low, "torch 사용 감지"
