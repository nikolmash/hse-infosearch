import os
import re
import json
import nltk
import pymorphy2
import collections
import numpy as np
from joblib import dump
from string import punctuation
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

# сформируем пути до каждой папки с субтитрами
curr_dir = os.getcwd()
subtitles_path = os.path.join(curr_dir, 'friends-data')
dirs = os.listdir(subtitles_path)

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


# обрабатываем каждый файл и добавляем в виде строки к будущему списку-корпусу
corpus = []
for dir in dirs:
    dirpath = os.path.join(subtitles_path, dir)
    files = os.listdir(dirpath)
    for file in files:
        fpath = os.path.join(dirpath, file)
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read()
            corpus.append(preprocessing(text))

# создаем матрицу term-document
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
# транспонированная матрица понадобится для удобства, чтобы извлекать нужную строчку для каждого слова
X_t = X.T
ii_dict = collections.defaultdict(list)
# для каждого слова берем необходимую строчку и добавляем в словарь id документов, где оно встречаются
for term in vectorizer.get_feature_names():
    row = X_t.getrow(vectorizer.vocabulary_.get(term)).toarray().ravel()
    ii_dict[term] = np.where(row > 0)[0].tolist()


# сохраняем матрицу term-document
dump(X, 'term-document.pkl')
# для удобства я сохраняю объект CountVectorizer, потому что 2 задание находится в ipynb тетрадке для наглядности
dump(vectorizer, 'vectorizer.pkl')
#сохраняю словарь в json файл
with open('inv_index.json', 'w', encoding='utf-8') as f:
    json.dump(ii_dict, f, ensure_ascii = False, indent = 4)
