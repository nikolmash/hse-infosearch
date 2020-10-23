import preprocessing
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

answers_df = pd.read_excel('answers_base.xlsx')
queries_df = pd.read_excel('queries_base.xlsx')
queries_df = queries_df[['Текст вопроса', 'Номер связки\n']].dropna()

queries_train, queries_test = train_test_split(queries_df, test_size=0.3, random_state=0)
documents = answers_df['Текст вопросов'].append(queries_train['Текст вопроса'], ignore_index=True)

documents_prep = documents.apply(lambda x: ' '.join(preprocessing.preprocessing(x)))
documents_ner = documents.apply(lambda x: ' '.join(preprocessing.preprocessing(preprocessing.preprocess_ner(x))))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents_prep)

vectorizer_ner = TfidfVectorizer()
X_ner = vectorizer_ner.fit_transform(documents_ner)


dump(X, 'text_representations/tfidf.pkl')
dump(vectorizer, 'text_representations/vectorizer.pkl')

dump(X_ner, 'text_representations/tfidf_ner.pkl')
dump(vectorizer_ner, 'text_representations/vectorizer_ner.pkl')