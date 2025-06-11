import os, re
import numpy as np
from nltk.cluster import kmeans
from sklearn.cluster import HDBSCAN
import gensim
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import requests
import nltk
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from natasha import NewsEmbedding, Segmenter, NewsSyntaxParser, Doc, MorphVocab, NewsMorphTagger
from sklearn.model_selection import GridSearchCV
from keybert import KeyBERT
from sklearn.cluster import KMeans
from gensim.models import Word2Vec
from bertopic import BERTopic

# Инициализация ресурсов NLTK
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
load_dotenv()
stop_words = stopwords.words("russian")

def get_groups_info(user_id, token):
    try:
        url = 'https://api.vk.com/method/users.getSubscriptions'
        params = {
            'user_id': user_id,
            'access_token': token,
            'extended': 1,
            'fields': 'description',
            'count': 200,
            'v': '5.131'
        }
        response = requests.get(url, params=params)
        return create_groups_dict(response.json())
    except Exception as e:
        print(f'Error of API: {e}')
        return None

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def create_groups_dict(data):
    groups = {}
    for group in data['response']['items']:
        name = group.get('name', 'Без названия')  # Если нет названия, будет возвращено 'Без названия'
        description = group.get('description', 'Нет описания')
        groups[name] = description
    return groups

def PrepareText(text, excludeWords = ""):
    eng = 'abcdefghijklmnopqrstuvwxyz'
    sentences = []
    tempSentence = ""
    tempWord = ""
    directSpeechFlag = False
    concatSentenceFlag = False
    for i in range(len(text)):
      text[i] = text[i].replace('!', '.').replace("%", ' ').replace("\xa0", ' ').replace('(', ' ').replace(')', ' ').replace('?', '.').replace('\n', ' ').replace("..", '.').replace("...", '.')

      letterWasFoundFlag = False

      for j in range(len(text[i])):
          if (len(text[i]) > j + 1 and text[i][j] == '-' and text[i][j + 1] == ' ' and letterWasFoundFlag):
              concatSentenceFlag = True
          if (text[i][j].lower() not in eng and text[i][j] != '.' and text[i][j] != ',' and text[i][j] != '\"' and text[i][j] != ":" and text[i][j] != "(" and text[i][j] != ")" and (text[i][j] != '-' or (len(text[i]) != j + 1 and text[i][j] == '-' and text[i][j + 1] != ' '))):
              letterWasFoundFlag = True
              tempWord += text[i][j]
          if (text[i][j] == '\"'):
              directSpeechFlag = not directSpeechFlag
          if (text[i][j] == ' '):
              if (len(tempWord) > 0 and tempWord.replace('.', '').replace(' ', '').replace(',', '') not in excludeWords):
                  if (concatSentenceFlag and len(tempSentence) == 0):
                      sentences[len(sentences) - 1] += tempWord
                  else:
                      tempSentence += tempWord
              tempWord = ""
          if (text[i][j] == '.'):
              if (len(tempWord) > 0 and tempWord.replace('.', '').replace(' ', '').replace(',', '') not in excludeWords):
                  if (concatSentenceFlag and len(tempSentence) == 0):
                      sentences[len(sentences) - 1] += tempWord
                  else:
                      tempSentence += tempWord
              tempWord = ""
              if (concatSentenceFlag):
                  letterWasFoundFlag = False
              concatSentenceFlag = False

              if (not directSpeechFlag and len(tempSentence) > 0):
                  sentences.append(tempSentence)
                  tempSentence = ""

    return sentences

def deEmojify(text):
    emoji_pattern = re.compile("["
      u"\U0001F600-\U0001F64F" # emoticons
      u"\U0001F300-\U0001F5FF" # symbols & pictographs
      u"\U0001F680-\U0001F6FF" # transport & map symbols
      u"\U0001F1E0-\U0001F1FF" # flags (iOS)
      u"\U00002702-\U000027B0"
      u"\U000024C2-\U0001F251"
      u"\U0001f926-\U0001f937"
      u'\U00010000-\U0010ffff'
      u"\u200d"
      u"\u2640-\u2642"
      u"\u2600-\u2B55"
      u"\u23cf"
      u"\u23e9"
      u"\u231a"
      u"\u3030"
      u"\ufe0f"
      u"\u2069"
      u"\u2066"
      u"\u200c"
      u"\u2068"
      u"\u2067"
          "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

class TextProcessing:
    def __init__(self, groups):
        self.groups = groups
        self.descriptions = list(self.groups.values())

    def preprocess_text(self):
        processed_texts = []
        emb = NewsEmbedding()
        segmenter = Segmenter()
        syntax_parser = NewsSyntaxParser(emb)
        morph_tagger = NewsMorphTagger(emb)
        morph_vocab = MorphVocab()

        docs = []

        # file = open("exclude.txt", encoding="UTF-8")

        self.descriptions = PrepareText(self.descriptions, excludeWords=stop_words)

        for desc in self.descriptions:
            if (len(desc) < 2):
                continue
            try:
                tempDocs = []
                doc = Doc(desc)
                doc.segment(segmenter)
                doc.parse_syntax(syntax_parser)
                doc.tag_morph(morph_tagger)
            except:
                pass

            for token in doc.tokens:
                token.lemmatize(morph_vocab)

            docs.append(doc)


        for doc in docs:
            text = ' '.join(x.lemma for x in doc.tokens if x.pos == 'NOUN')
            cleaned_text = self.clean_text(text)
            tokens = self.tokenization(cleaned_text)

            tokens = [w for w in tokens if not w.lower() in stop_words and "http" not in w.lower()]

            lemmatized_tokens = self.lemmatize(tokens)
            processed_texts.append(lemmatized_tokens)
        return processed_texts

    def clean_text(self, text):
        # Переводим текст в нижний регистр
        text = text.lower()
        text = re.sub(r"<.*?>", "", text)  # Удаление HTML тегов
        text = re.sub(r"[^a-zA-Zа-яА-Я\s]", "", text)  # Удаление всех символов, кроме букв
        text = re.sub(r"\s+", " ", text).strip()  # Удаление лишних пробелов
        return text

    def tokenization(self, text):
        return word_tokenize(text)

    def lemmatize(self, words):
        return [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]



class TextModeling:

    def __init__(self, texts):
        self.texts = texts


    def tfidf_keywords(self):
        joined_texts = [" ".join(text) for text in self.texts]

        tfidf_vectorizer = TfidfVectorizer(max_df=2, min_df=1, stop_words=stop_words, max_features=1000)
        tfidf_matrix = tfidf_vectorizer.fit_transform(joined_texts)
        feature_names = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = tfidf_matrix.sum(axis=0).A1

        keywords = sorted(zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True)
        return [kw[0] for kw in keywords]

    def keybert_words(self):
        joined_texts = [" ".join(text) for text in self.texts]
        kb_model = KeyBERT()
        keywords = kb_model.extract_keywords(joined_texts)

        return keywords

    def bert_topic(self):
        from collections import Counter

        joined_texts = [" ".join(text) for text in self.texts if text]


        docs = list(set([t.strip() for t in joined_texts if t.strip()]))

        if not docs:
            print("Нет текстов для моделирования тем.")
            return None


        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

        try:

            topic_model = BERTopic(embedding_model=embedding_model, verbose=False)
            topics, _ = topic_model.fit_transform(docs)


            topics_clean = [t for t in topics if t != -1]
            topic_freq = Counter(topics_clean)

            total = sum(topic_freq.values())
            profile = {topic_id: round(freq / total, 3) for topic_id, freq in topic_freq.items()}

            topic_descriptions = {}
            for topic_id in profile:
                words = topic_model.get_topic(topic_id)
                phrase = " ".join([w for w, _ in words[:5]])
                topic_descriptions[topic_id] = phrase

            return {
                "topic_profile": profile,
                "topic_descriptions": topic_descriptions
            }

        except Exception as e:
            print(f"Ошибка в BERTopic: {e}")
            return None



load_dotenv()


groups = get_groups_info(140716441, os.getenv('TOKEN'))
t_proc = TextProcessing(groups)
processed_text = t_proc.preprocess_text()

model = TextModeling(processed_text)
kw1 = model.keybert_words()[:10]
topics1= model.bert_topic()
print(f'Topics {topics1}')

groups2 = get_groups_info(140716441, os.getenv('TOKEN'))
t_proc1 = TextProcessing(groups)
processed_text2 = t_proc.preprocess_text()
#
# model = TextModeling(processed_text)
# kw2 = model.keybert_words()[:10]
# topics2 = model.bert_topic()



