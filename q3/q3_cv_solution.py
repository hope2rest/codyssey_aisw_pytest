import numpy as np
import json
import os
import pathlib
from PIL import Image
from scipy.ndimage import label as scipy_label, binary_closing

THRESHOLD = 30
MIN_AREA = 100

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
GAUSS3 = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0


def conv2d(image, kernel):
    """2D 합성곱 (valid 모드, NumPy 순수 구현)"""
    kH, kW = kernel.shape
    iH, iW = image.shape
    oH, oW = iH - kH + 1, iW - kW + 1
    k_flip = kernel[::-1, ::-1]
    shape = (oH, oW, kH, kW)
    strides = (image.strides[0], image.strides[1], image.strides[0], image.strides[1])
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    return np.einsum("ijkl,kl->ij", windows, k_flip)


def _pad_to(arr, h, w):
    ph, pw = h - arr.shape[0], w - arr.shape[1]
    return np.pad(arr, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2)), mode="edge")


def _to_gray(rgb):
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def count_boxes(image_path, threshold=THRESHOLD, min_area=MIN_AREA):
    """박스 카운팅 파이프라인"""
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)
    gray = _to_gray(rgb)
    h, w = gray.shape

    blurred = _pad_to(conv2d(gray, GAUSS3), h, w)
    Gx = conv2d(blurred, SOBEL_X)
    Gy = conv2d(blurred, SOBEL_Y)
    magnitude = _pad_to(np.sqrt(Gx**2 + Gy**2), h, w)

    binary = (magnitude > threshold).astype(np.uint8)
    closed = binary_closing(binary, structure=np.ones((3, 3), dtype=np.uint8), iterations=3)
    labeled, num = scipy_label(closed)

    count = 0
    for cid in range(1, num + 1):
        if int(np.sum(labeled == cid)) >= min_area:
            count += 1
    return count


def main():
    image_dir = "data/images"
    label_path = "data/labels.json"

    with open(label_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    # 실제 이미지 파일이 있는 것만 처리 (test_ 접두사 제외)
    image_names = sorted(k for k in labels if not k.startswith("test"))

    predictions = {}
    for name in image_names:
        img_path = os.path.join(image_dir, name + ".png")
        if not os.path.exists(img_path):
            continue
        predictions[name] = count_boxes(img_path)

    # 카테고리별 메트릭
    metrics = {}
    for cat in ["easy", "medium", "hard"]:
        keys = sorted(k for k in labels if k.startswith(cat + "_") and k in predictions)
        if not keys:
            metrics[cat] = {"mae": 0.0, "accuracy": 0.0}
            continue
        errors = [abs(predictions[k] - labels[k]) for k in keys]
        mae = float(np.mean(errors))
        acc = float(sum(1 for e in errors if e == 0) / len(errors))
        metrics[cat] = {"mae": round(mae, 4), "accuracy": round(acc, 4)}

    # worst case (hard 카테고리)
    hard_keys = [k for k in labels if k.startswith("hard_") and k in predictions]
    worst_case = max(hard_keys, key=lambda k: abs(predictions[k] - labels[k])) if hard_keys else ""

    failure_reasons = [
        "박스들이 밀집하거나 서로 겹쳐 있을 경우 Sobel 엣지가 연결되어 여러 박스가 하나의 연결 컴포넌트로 병합되므로, 규칙 기반 카운팅은 실제 개수를 심각하게 과소 추정한다.",
        "적재(Stacked) 형태나 불규칙한 다각형 형태에서는 단일 고정 임계값과 2D 엣지만으로 박스 경계를 올바르게 분리할 수 없으며, 깊이 정보 없이는 앞뒤 박스를 구분하기 불가능하다.",
        "크기 편차가 매우 큰 환경에서는 하나의 고정 min_area 값으로 소형 박스(노이즈와 유사)와 대형 박스를 동시에 처리할 수 없어 소형 박스가 노이즈로 오인되어 필터링된다.",
        "조명 불균일, 그림자, 박스 표면 질감에 의해 박스 내부에도 강한 엣지가 생성되어 단일 박스가 여러 컴포넌트로 분리되거나, 배경 텍스처가 박스로 오인식되는 위양성이 발생한다.",
    ]

    why_learning_based = (
        "규칙 기반 방법은 고정 임계값과 단순 형태 분석에 의존하므로 조명 변화, "
        "박스 겹침, 크기 편차, 적재 구조 등 복잡한 실세계 조건에 일반화할 수 없다. "
        "CNN 등 학습 기반 모델은 대규모 데이터로부터 특징을 자동 학습하여 "
        "다양한 환경에서도 강인한 객체 탐지가 가능하다."
    )

    result = {
        "predictions": predictions,
        "metrics": metrics,
        "worst_case_image": worst_case,
        "failure_reasons": failure_reasons,
        "why_learning_based": why_learning_based,
    }

    pathlib.Path("result_q3.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
