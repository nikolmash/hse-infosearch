import numpy as np


def bm25_ranging(query, all_words, base):
    # предобрабатываем и преобразуем в вектор из 0 и 1 запрос
    new_doc = np.array([1 if word in query else 0 for word in all_words])
    # умножние матрицы на вектор и сортировка по убыванию
    print(base.shape, new_doc.shape)
    scoring = enumerate(base.dot(new_doc.T))
    sorted_docs = sorted(scoring, key=lambda x: x[1], reverse=True)
    indexes = [x[0] for x in sorted_docs[:10]]
    return indexes