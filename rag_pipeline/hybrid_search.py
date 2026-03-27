from rank_bm25 import BM25Okapi

documents = []
tokenized_corpus = []

def build_bm25(docs):
    global documents, tokenized_corpus

    documents = docs
    tokenized_corpus = [
        doc.page_content.split() for doc in docs
    ]

    return BM25Okapi(tokenized_corpus)


def keyword_search(query, bm25, top_k=3):
    tokenized_query = query.split()
    scores = bm25.get_scores(tokenized_query)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return [doc for doc, _ in ranked]