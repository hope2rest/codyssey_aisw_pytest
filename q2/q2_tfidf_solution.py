import re
import json
import pathlib
import unicodedata
import numpy as np
from collections import Counter


def preprocess(text, stopwords):
    """NFC 정규화 → 소문자 → 특수문자 제거 → 토큰화 → 불용어 제거 → 1자 제거"""
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[^가-힣a-z0-9\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords and len(t) > 1]
    return tokens


def cosine_similarity(a, b):
    """두 벡터 간 코사인 유사도 계산"""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def search(query, documents, vocab, idf, stopwords, top_k=3):
    """쿼리에 대해 TF-IDF 코사인유사도 기반 상위 top_k 문서 검색"""
    w2i = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    N = len(documents)

    # 쿼리 TF-IDF
    q_tokens = preprocess(query, stopwords)
    q_tf = np.zeros(V)
    if q_tokens:
        cnt = Counter(q_tokens)
        for w, c in cnt.items():
            if w in w2i:
                q_tf[w2i[w]] = c / len(q_tokens)
    q_tfidf = q_tf * idf

    # 문서별 TF-IDF 계산 및 유사도
    sims = []
    for i, doc_tokens in enumerate(documents):
        d_tf = np.zeros(V)
        if doc_tokens:
            cnt = Counter(doc_tokens)
            for w, c in cnt.items():
                if w in w2i:
                    d_tf[w2i[w]] = c / len(doc_tokens)
        d_tfidf = d_tf * idf
        sim = cosine_similarity(q_tfidf, d_tfidf)
        sims.append((i, sim))

    sims.sort(key=lambda x: (-x[1], x[0]))
    top = sims[:top_k]

    if all(v == 0 for _, v in top):
        return [{"doc_index": i, "similarity": 0.0} for i in range(top_k)]
    return [{"doc_index": i, "similarity": round(v, 6)} for i, v in top]


def main():
    data_dir = "data"

    # 불용어 로드
    with open(f"{data_dir}/stopwords.txt", "r", encoding="utf-8") as f:
        stopwords = set(line.strip() for line in f if line.strip())

    # 문서 로드 (빈 줄 제외)
    with open(f"{data_dir}/documents.txt", "r", encoding="utf-8") as f:
        raw_docs = [line.strip() for line in f if line.strip()]

    # 쿼리 로드
    with open(f"{data_dir}/queries.txt", "r", encoding="utf-8") as f:
        queries = [line.strip() for line in f if line.strip()]

    N = len(raw_docs)

    # 전체 문서 전처리
    tokenized = [preprocess(doc, stopwords) for doc in raw_docs]

    # 어휘 사전 구축 (정렬)
    vocab = sorted(set(w for doc in tokenized for w in doc))
    V = len(vocab)
    w2i = {w: i for i, w in enumerate(vocab)}

    # TF 행렬
    tf = np.zeros((N, V))
    for i, tokens in enumerate(tokenized):
        if not tokens:
            continue
        cnt = Counter(tokens)
        for w, c in cnt.items():
            tf[i, w2i[w]] = c / len(tokens)

    # IDF: log((N+1)/(df+1)) + 1
    df_v = np.sum(tf > 0, axis=0)
    idf = np.log((N + 1) / (df_v + 1)) + 1

    # TF-IDF 행렬
    tfidf = tf * idf

    # 각 쿼리별 검색
    search_results = []
    for q in queries:
        top3 = search(q, tokenized, vocab, idf, stopwords)
        search_results.append({"query": q, "top3": top3})

    # 결과 저장
    result = {
        "num_documents": N,
        "vocab_size": V,
        "tfidf_matrix_shape": [N, V],
        "search_results": search_results,
    }

    pathlib.Path("result_q2.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
