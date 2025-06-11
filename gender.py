import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.tokenize import word_tokenize
from pymorphy3 import MorphAnalyzer
import time
import os
from dotenv import load_dotenv

nltk.download('punkt')
morph = MorphAnalyzer()


def get_users(token, ids):
    url = "https://api.vk.com/method/users.get"
    users = []

    for id in ids[:300]:
        params = {
            'access_token': token,
            'v': '5.131',
            'fields': 'sex',
            'user_id': id
        }

        try:
            response = requests.get(url, params=params)
            data = response.json()

            if 'error' in data:
                print(f"Ошибка VK API: {data['error']['error_msg']}")

            users_list = data.get('response', [])
            for user in users_list:
                users.append({
                    'id': user['id'],
                    'gender': user.get('sex', None)
                })
                print(f'User_id:{user['id']}')
            time.sleep(0.34)

        except Exception as e:
            print(f"Ошибка при выполнении запроса: {e}")

    return pd.DataFrame(users)

def clean_tokenize_lemmatize(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^а-яА-Я ]", " ", text)
    tokens = word_tokenize(text, language='russian')
    lemmas = [morph.parse(token)[0].normal_form for token in tokens if len(token) > 2]
    print(lemmas)
    return " ".join(lemmas)

# === Получение описаний групп пользователя ===
def get_group_descriptions(user_id):
    url = "https://api.vk.com/method/groups.get"
    params = {
        "access_token": os.getenv('TOKEN'),
        "v": '5.131',
        "user_id": user_id,
        "extended": 1,
        "fields": "description",
        "count": 200
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if "response" in data:
            groups = data["response"]["items"]
            descriptions = [group.get("description", "") for group in groups if group.get("description")]
            combined = " ".join(descriptions)
            return clean_tokenize_lemmatize(combined)
        else:
            return ""
    except Exception as e:
        print(f"Error for user {user_id}: {e}")
        return ""

# === Основной процессинг DataFrame ===
def process_user_dataframe(df):
    group_texts = []
    for user_id in df["id"]:
        print(f'user_id: {user_id}')
        text = get_group_descriptions(user_id)
        print(text)
        group_texts.append(text)
        time.sleep(0.2)
    df["group_text"] = group_texts
    return df



load_dotenv()
ids = pd.read_csv('dataset/vk_dataset.csv', sep=',')
df = get_users(os.getenv('TOKEN'), ids['id'].tolist())

df = process_user_dataframe(df)
df.to_csv("dataset/gender_vk.csv", index=False, mode='a')
