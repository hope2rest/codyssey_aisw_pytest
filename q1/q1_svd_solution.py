import numpy as np
import pandas as pd
import json
import pathlib


def main():
    # 1. 데이터 로드 (헤더 없음, 500x100)
    df = pd.read_csv("data/sensor_data.csv", header=None)
    X_raw = df.values.astype(float)

    # 2. NaN 처리: 열별 평균으로 대체
    col_means = np.nanmean(X_raw, axis=0)
    nan_mask = np.isnan(X_raw)
    X_raw[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    # 3. 표준화 (ddof=0, 모집단 표준편차)
    mean = np.mean(X_raw, axis=0)
    std = np.std(X_raw, axis=0, ddof=0)

    # 상수열(std≈0) 처리: 1로 대체하여 0으로 만듦
    std_safe = np.where(std == 0, 1.0, std)
    X = (X_raw - mean) / std_safe

    # 4. SVD 분해
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    # 5. 설명된 분산 비율
    evr = S**2 / np.sum(S**2)

    # 6. 최적 k (누적 분산 >= 95%)
    cum = np.cumsum(evr)
    optimal_k = int(np.argmax(cum >= 0.95) + 1)
    cumulative_variance_at_k = round(float(cum[optimal_k - 1]), 6)

    # 7. 복원 및 MSE 계산
    X_reconstructed = (U[:, :optimal_k] * S[:optimal_k]) @ Vt[:optimal_k, :]
    mse = round(float(np.mean((X - X_reconstructed) ** 2)), 6)

    # 8. 결과 저장
    result = {
        "optimal_k": optimal_k,
        "cumulative_variance_at_k": cumulative_variance_at_k,
        "reconstruction_mse": mse,
        "top_5_singular_values": [round(float(s), 6) for s in S[:5]],
        "explained_variance_ratio_top5": [round(float(r), 6) for r in evr[:5]],
    }

    pathlib.Path("result_q1.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
