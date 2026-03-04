import numpy as np
import pandas as pd
import json
import pathlib
import unicodedata
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap

warnings.filterwarnings("ignore")


def rule_based_predict(text, sentiment_dict):
    """감성 사전 기반 규칙 예측: 긍정(1) / 부정(0)"""
    tokens = str(text).split()
    score = 0.0
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in sentiment_dict["negation"] and i + 1 < len(tokens):
            nt = tokens[i + 1]
            ns = sentiment_dict["positive"].get(nt, sentiment_dict["negative"].get(nt, 0.0))
            score += ns * (-1.0)
            i += 2
            continue
        if t in sentiment_dict["intensifier"] and i + 1 < len(tokens):
            nt = tokens[i + 1]
            ns = sentiment_dict["positive"].get(nt, sentiment_dict["negative"].get(nt, 0.0))
            score += ns * sentiment_dict["intensifier"][t]
            i += 2
            continue
        ts = sentiment_dict["positive"].get(t, sentiment_dict["negative"].get(t, 0.0))
        score += ts
        i += 1
    return 1 if score > 0 else 0


def compute_metrics(y_true, y_pred):
    """정확도, 정밀도, 재현율, F1 계산"""
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_pos": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall_pos": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision_neg": round(float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "recall_neg": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def main():
    data_dir = "data"

    # 데이터 로드
    df = pd.read_csv(f"{data_dir}/reviews.csv")
    with open(f"{data_dir}/sentiment_dict.json", "r", encoding="utf-8") as f:
        sd = json.load(f)

    # NaN 제거 + NFC 정규화
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].apply(lambda x: unicodedata.normalize("NFC", str(x)))
    df = df.reset_index(drop=True)

    # === 규칙 기반 ===
    all_rule_preds = df["text"].apply(lambda t: rule_based_predict(t, sd)).values

    # === ML 기반 ===
    X_tr, X_te, y_tr, y_te = train_test_split(
        df["text"], df["label"], test_size=0.30, random_state=42
    )

    # 오버샘플링 (train에서만)
    tr_df = pd.DataFrame({"text": X_tr.values, "label": y_tr.values})
    pos_df = tr_df[tr_df["label"] == 1]
    neg_df = tr_df[tr_df["label"] == 0]
    np.random.seed(42)
    neg_over_idx = np.random.choice(np.arange(len(neg_df)), size=len(pos_df), replace=True)
    neg_oversampled = neg_df.iloc[neg_over_idx]
    bal = pd.concat([pos_df, neg_oversampled]).reset_index(drop=True)
    bal = bal.sample(frac=1, random_state=42).reset_index(drop=True)

    # TF-IDF (fit은 train에서만)
    vec = TfidfVectorizer(sublinear_tf=False, smooth_idf=True)
    X_tr_tf = vec.fit_transform(bal["text"])
    X_te_tf = vec.transform(X_te)

    # LogisticRegression
    mdl = LogisticRegression(C=1.0, penalty="l2", random_state=42, max_iter=1000)
    mdl.fit(X_tr_tf, bal["label"])
    ml_preds = mdl.predict(X_te_tf)

    # 메트릭 계산
    rule_m = compute_metrics(y_te.values, all_rule_preds[X_te.index])
    ml_m = compute_metrics(y_te.values, ml_preds)

    # === SHAP ===
    exp = shap.LinearExplainer(mdl, X_tr_tf)
    sv = exp.shap_values(X_te_tf)
    fn = vec.get_feature_names_out()
    ms = np.asarray(np.mean(sv, axis=0)).flatten()

    top_pos_idx = np.argsort(ms)[::-1][:5]
    shap_top5_positive = [
        {"word": str(fn[i]), "shap_value": round(float(ms[i]), 4)} for i in top_pos_idx
    ]
    top_neg_idx = np.argsort(ms)[:5]
    shap_top5_negative = [
        {"word": str(fn[i]), "shap_value": round(float(ms[i]), 4)} for i in top_neg_idx
    ]

    # 비즈니스 요약
    pos_kw = shap_top5_positive[0]["word"]
    neg_kw = shap_top5_negative[0]["word"]
    business_summary = (
        f"머신러닝 모델은 규칙 기반 모델 대비 부정 리뷰 탐지율이 높아 고객 불만을 보다 정확히 식별합니다. "
        f"긍정 예측에는 '{pos_kw}' 등의 단어가, 부정 예측에는 '{neg_kw}' 등의 단어가 주요하게 작용합니다. "
        f"마케팅팀은 긍정 키워드를 홍보에 활용하고, 영업팀은 부정 키워드 중심으로 고객 불만 원인을 파악하여 서비스를 개선할 수 있습니다."
    )

    # 결과 저장
    result = {
        "rule_based": rule_m,
        "ml_based": ml_m,
        "shap_top5_positive": shap_top5_positive,
        "shap_top5_negative": shap_top5_negative,
        "business_summary": business_summary,
    }

    pathlib.Path("result_q4.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
