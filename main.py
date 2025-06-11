import requests
import re
import pandas as pd
import numpy as np
from datetime import datetime
from time import sleep
import os
from dotenv import load_dotenv
from scipy.stats import iqr
import json


def get_users(token):
    all_users = []
    pattern = re.compile(r'^\d{1,2}\.\d{1,2}\.\d{4}$')

    url = "https://api.vk.com/method/users.search"
    params = {
        'access_token': token,
        'v': '5.131',
        'count': 200,
        'fields': 'bdate',
        'age_from': 30
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()
        with open('dataset/s.json', 'r', encoding='utf-8') as f:  # <- добавьте encoding='utf-8'
            data = json.load(f)

        df = pd.DataFrame(data['response']['items'])[["id", "bdate"]]

        if 'error' in data:
            print(f"Ошибка VK API: {data['error']['error_msg']}")
            return pd.DataFrame()

        users = data['response']['items']

        for user in users:
            if 'bdate' in user and pattern.match(user['bdate']):
                day, month, year = map(int, user['bdate'].split('.'))
                age = datetime.now().year - year
                if age >= 100:
                    continue

                all_users.append({
                    'id': user['id'],
                    'bdate': user['bdate'],
                    'day': day,
                    'month': month,
                    'year': year,
                    'age': age
                })

        df = pd.DataFrame(all_users)

        return df

    except Exception as e:
        print(f"Ошибка при выполнении запроса: {e}")
        return pd.DataFrame()


def get_users_with_friends_stats(token, user_ids):

    friends_stats = []

    for user_id in user_ids:
        try:
            print(f'Обработка пользователя {user_id}')
            friends_params = {
                'access_token': token,
                'v': '5.131',
                'user_id': user_id,
                'fields': "bdate"
            }

            sleep(1.5)
            response = requests.get('https://api.vk.com/method/friends.get', params=friends_params)
            data = response.json()
            print(f'RESPONSE: {response}')


            if 'error' in data or 'response' not in data:
                print(f"Ошибка VK API: {data['error']['error_msg']}")
                friends_stats.append({
                    'id': user_id,
                    'friends_count': np.nan,
                    'friends_avg_age': np.nan,
                    'friends_median_age': np.nan,
                    'friends_std_age': np.nan,
                    'friends_mode_age': np.nan,
                    'friends_child': np.nan,
                    'friends_stud': np.nan,
                    'friends_young': np.nan,
                    'friends_mature': np.nan,
                    'friends_old': np.nan
                })
                continue

            friends = data['response']['items']
            ages = []
            pattern = re.compile(r'^\d{1,2}\.\d{1,2}\.\d{4}$')

            for friend in friends:
                if 'bdate' in friend and pattern.match(friend['bdate']):
                    day, month, year = map(int, friend['bdate'].split('.'))
                    age = datetime.now().year - year
                    if age >= 100:
                        continue
                    ages.append(age)
            print(f'Кол-во друзей: {len(ages)}')

            if len(ages) == 0: continue

            print(f'child: {len([a for a in ages if 14 <= a <= 17]) / len(ages)}')
            print(f'young_count: {len([a for a in ages if 18 <= a <= 35]) / len(ages)}')
            print(f'middle_count: {len([a for a in ages if 36 <= a <= 55]) / len(ages)}')
            print(f'mature_count: {len([a for a in ages if 56 <= a <= 65]) / len(ages)}')
            print(f'old_count: {len([a for a in ages if a >= 65]) / len(ages)}')
            stats = {
                'id': user_id,
                'friends_count': len(friends) if ages else 0,
                'friends_ages': len(ages) if ages else 0,
                'avg_age': np.mean(ages) if ages else 0,
                'median_age': np.median(ages) if ages else 0,
                'std_age': np.std(ages) if ages else 0,
                'mode_age': pd.Series(ages).mode()[0] if ages else 0,
                's_range': np.max(ages) - np.min(ages) if len(ages) > 0 else 0,
                'iqr': iqr(ages) if len(ages) > 0 else 0,
                'var': np.var(ages) if len(ages) > 0 else 0,
                'child_count': len([a for a in ages if 14 <= a <= 17]) / len(ages) if ages else 0,
                'young_count': len([a for a in ages if 18 <= a <= 35]) / len(ages) if ages else 0,
                'middle_count': len([a for a in ages if 36 <= a <= 55]) / len(ages)if ages else 0,
                'mature_count': len([a for a in ages if 56 <= a <= 65]) / len(ages) if ages else 0,
                'old_count': len([a for a in ages if a >= 65]) / len(ages) if ages else 0
            }
            print(stats)
            friends_stats.append(stats)

        except Exception as e:
            print(f"Ошибка для пользователя {user_id}: {str(e)}")
            continue

    return pd.DataFrame(friends_stats)

load_dotenv()

df = get_users(os.getenv('TOKEN'))

if not df.empty:
    friends_stats = get_users_with_friends_stats(os.getenv('TOKEN'), df['id'].tolist())
    if friends_stats is not None:
        df = df.merge(friends_stats, on='id', how='left')
        df.to_csv('dataset/test_file.csv', mode='a', index=False)
