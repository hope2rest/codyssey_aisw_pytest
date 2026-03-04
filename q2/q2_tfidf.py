def preprocess(text, stopwords):
    """텍스트 전처리"""
    # TODO: 구현
    pass


def cosine_similarity(a, b):
    """두 벡터 간 코사인 유사도 계산"""
    # TODO: 구현
    pass


def search(query, documents, vocab, idf, stopwords, top_k=3):
    """쿼리에 대해 상위 top_k 문서 검색"""
    # TODO: 구현
    pass


def main():
    data_dir = "data"

    # TODO: 문서 로드 (documents.txt)

    # TODO: 불용어 로드 (stopwords.txt)

    # TODO: 쿼리 로드 (queries.txt)

    # TODO: 전체 문서 전처리

    # TODO: 어휘 사전 구축

    # TODO: IDF 계산

    # TODO: TF-IDF 행렬 생성

    # TODO: 각 쿼리에 대해 검색 수행

    result = {
        "num_documents": None,
        "vocab_size": None,
        "tfidf_matrix_shape": [],
        "search_results": [],
    }

    return result


if __name__ == "__main__":
    import json
    print(json.dumps(main(), ensure_ascii=False, indent=2))
