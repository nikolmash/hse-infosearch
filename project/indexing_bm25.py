import preprocessing
import numpy as np
import pandas as pd
from math import log
from joblib import load
from sklearn.model_selection import train_test_split

answers_df = pd.read_excel('answers_base.xlsx')
queries_df = pd.read_excel('queries_base.xlsx')
queries_df = queries_df[['Текст вопроса', 'Номер связки\n']].dropna()

queries_train, queries_test = train_test_split(queries_df, test_size=0.3, random_state=0)
documents = answers_df['Текст вопросов'].append(queries_train['Текст вопроса'], ignore_index=True)

documents_prep = documents.apply(lambda x: ' '.join(preprocessing.preprocessing(x)))
documents_ner = documents.apply(lambda x: ' '.join(preprocessing.preprocessing(preprocessing.preprocess_ner(x))))

vectorizer = load('text_representations/vectorizer.pkl')
vectorizer_ner = load('text_representations/vectorizer_ner.pkl')


def bm25(term, document, k=2.0, b=0.75) -> float:
    doc = np.array(document.split())
    length = doc.shape[0]
    tf = doc[doc == term].shape[0]/length
    a = np.array([term in d for d in documents])
    nq = a[a==True].shape[0]
    idf = log((N-nq+0.5)/(nq+0.5))
    return idf*tf*(k+1)/(tf+k*(1-b+b*length/avg_length))


def bm25_base_creating():
    # сначала создаем пустую матрицу размера (кол-во документов, длина словаря)
    base = np.zeros((N, len(all_words)))
    # заполняем ее значениями bm25 для каждого слова словаря и каждого документа коллекции
    for i, doc in enumerate(documents):
        base[i,:] = np.array([bm25(word, doc) for word in all_words])
    return base

N = documents.shape[0]
avg_length = np.mean([len(x.split()) for x in documents_prep])
all_words = vectorizer.get_feature_names()
bm25_base = bm25_base_creating()
np.save('bm25.npy', bm25_base)

avg_length = np.mean([len(x.split()) for x in documents_ner])
all_words = vectorizer_ner.get_feature_names()
bm25_base_ner = bm25_base_creating()
np.save('bm25.npy', bm25_base_ner)