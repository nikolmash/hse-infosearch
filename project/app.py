import time
import numpy as np
import pandas as pd
import preprocessing
from flask import Flask, request, render_template
from joblib import load
from sklearn.model_selection import train_test_split
from tfidf_search import tf_idf_ranging
from bm25_search import bm25_ranging

answers_df = pd.read_excel('answers_base.xlsx')
queries_df = pd.read_excel('queries_base.xlsx')
queries_df = queries_df[['Текст вопроса', 'Номер связки\n']].dropna()
queries_train, _ = train_test_split(queries_df, test_size=0.3, random_state=0)
documents = answers_df['Текст вопросов'].append(queries_train['Текст вопроса'], ignore_index=True)
answers = answers_df['Номер связки'].append(queries_train['Номер связки\n'], ignore_index=True)

vec = load('text_representations/vectorizer.pkl')
vec_ner = load('text_representations/vectorizer_ner.pkl')

tf_idf = load('text_representations/tfidf.pkl')
tf_idf_ner = load('text_representations/tfidf_ner.pkl')
bm25 = np.load('text_representations/bm25.npy')
bm25_ner = np.load('text_representations/bm25_ner.npy')


app = Flask(__name__)


def search(query):
    if request.form['method'] == 'tf-idf':
        prep_query = ' '.join(preprocessing.preprocessing(query))
        X = tf_idf
        vectorizer = vec
        if 1 in request.form.getlist('mycheckbox'):
            X = tf_idf_ner
            vectorizer = vec_ner
        return tf_idf_ranging(prep_query, vectorizer, X)
    if request.form['method'] == 'bm25':
        prep_query = ' '.join(preprocessing.preprocessing(query))
        base = bm25
        all_words = vec_ner.get_feature_names()
        if 1 in request.form.getlist('mycheckbox'):
            base = bm25
            all_words = vec_ner.get_feature_names()
        return bm25_ranging(prep_query, all_words, base)
    else:
        raise ValueError('Метод поиска не выбран')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/results', methods=['POST'])
def ranging():
    start = time.time()
    indexes = search(request.form['query'])
    ans = answers_df[answers_df['Номер связки'].isin(answers[indexes].values)]['Текст ответа']
    end = time.time()
    return render_template('results.html',
                           query=request.form['query'],
                           docs=zip(documents[indexes], ans),
                           time=end-start
    )



if __name__ == '__main__':
    app.run(debug=True)

