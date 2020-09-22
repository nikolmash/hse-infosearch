import re
import nltk
import pymorphy2
import pandas as pd
import numpy as np
from math import log
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


# подготовим объекты для препроцессинга: лист стоп-слов, лист пунктуационных символов,
# и объект из pymorphy2 для морфологического анализа
nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()
stopwords = set(stopwords.words('russian'))
punkt = punctuation + '«»—…“”*№–'


# функция препроцессинга
def preprocessing(text):
    clean_text = re.sub('(\d|[a-zA-Z])*', '', text)
    clean_text = ' '.join([word.strip(punkt) for word in clean_text.lower().split()])
    clean_text = ' '.join([word for word in clean_text.split() if word not in stopwords])
    clean_text = ' '.join([morph.parse(word)[0].normal_form for word in clean_text.split()])
    return clean_text


# открываем таблицу с вопросами, совершаем над ними предобработку и составляем матрицу tf-idf
answers_df = pd.read_excel('answers_base.xlsx ')
documents = answers_df['Текст вопросов'].apply(preprocessing).values
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)


# функция ранжирования по tf-idf
def tf_idf_ranging(query):
    # предобрабатываем запрос и преобразуем его в вектор
    prep_query = preprocessing(query)
    new_doc = vectorizer.transform(prep_query).toarray()[0]
    # умножение матрицы на вектор и сортировка по убыванию
    scoring = enumerate(X.dot(new_doc.T))
    return sorted(scoring, key=lambda x: x[1], reverse=True)


# реализуем функцию для bm25
k = 2.0
b = 0.75
avg_length = np.mean([len(x.split()) for x in documents])
N = documents.shape[0]
all_words = vectorizer.get_feature_names()


def bm25(term, document) -> float:
    doc = np.array(document.split())
    length = doc.shape[0]
    tf = doc[doc == term].shape[0]/length
    a = np.array([term in d for d in documents])
    nq = a[a==True].shape[0]
    idf = log((N-nq+0.5)/(nq+0.5))
    return idf*tf*(k+1)/(tf+k*(1-b+b*length/avg_length))


# функция, создающая матрицу bm25
def bm25_base_creating():
    # сначала создаем пустую матрицу размера (кол-во документов, длина словаря)
    base = np.zeros((N, len(all_words)))
    # заполняем ее значениями bm25 для каждого слова словаря и каждого документа коллекции
    for i, doc in enumerate(documents):
        base[i,:] = np.array([bm25(word, doc) for word in all_words])
    return base


# функция ранжирования по bm25
def bm25_ranging(query):
    # создаем базу
    base = bm25_base_creating()
    # предобрабатываем и преобразуем в вектор из 0 и 1 запрос
    prep_query = preprocessing(query)
    new_doc = np.array([1 if word in prep_query else 0 for word in all_words])
    # умножние матрицы на вектор и сортировка по убыванию
    scoring = enumerate(base.dot(new_doc.T))
    return sorted(scoring, key=lambda x: x[1], reverse=True)
