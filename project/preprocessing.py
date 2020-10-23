import re
import nltk
import pymorphy2
from razdel import tokenize
from string import punctuation
from nltk.corpus import stopwords
from natasha import NewsNERTagger, NewsEmbedding, Doc, Segmenter


emb = NewsEmbedding()
segmenter = Segmenter()
ner_tagger = NewsNERTagger(emb)

nltk.download('stopwords')
morph = pymorphy2.MorphAnalyzer()
stopwords = set(stopwords.words('russian'))
punkt = punctuation + '«»—…“”*№–'


def tokenizing(text):
  """Функция, делящая текст на токены"""
  tokens = list(tokenize(text))
  return [_.text.lower() for _ in tokens]


def preprocessing(text):
  """Основная предобработка: удаление стоп-слов, запятых и лемматизация"""
  tokens = tokenizing(text)
  clean_text = [x for x in tokens if x not in stopwords and x not in punkt]
  clean_text = [morph.parse(word)[0].normal_form for word in clean_text if re.match('\W+', word) is None]
  return clean_text


def preprocess_ner(text):
    """Удаление именованных сущностей """
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    new_text = text
    for entity in doc.spans:
        new_text = new_text.replace(text[entity.start:entity.stop], '')
    return new_text