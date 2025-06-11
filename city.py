import os
import requests
import pandas as pd
import json
import numpy as np
from dotenv import load_dotenv
from collections import Counter


def evaluate_scores(ages_true, ages_pred, df=pd.DataFrame()):
    accuracy_scores = []
    for i in range(len(ages_true)):
        accuracy_scores.append(ages_pred == ages_true)

    df['age_true'] = ages_true
    df['age_calc'] = ages_pred
    df['acc'] = accuracy_scores

    return df

class CityPredictor:

    def __init__(self, token, user_id, Wuser=1, Wcomm=1):
        self.token = token
        self.Wuser, self.Wcomm = Wuser, Wcomm
        self.user_id = user_id

        self.friends = self.get_friends()
        self.groups = self.get_groups()


    def get_friends(self):
        try:
            params = {
                'access_token': self.token,
                'v': '5.131',
                'user_id': self.user_id,
                'fields': "city"
            }

            response = requests.get('https://api.vk.com/method/friends.get', params=params)
            return response.json()

        except Exception as e:
            print(f'Ошибка выполнения запроса: {e}')
            return None


    def create_distr_friends(self, data):
        friends = []
        for friend in data['response']['items']:
            if 'city' in friend and friend['city']:
                friends.append(friend['city']['title'])

        return dict(Counter(friends))



    def get_groups(self):
        try:
            params = {
                'access_token': self.token,
                'v': '5.131',
                'user_id': self.user_id,
                'extended': 1,
                'fields': "city"
            }

            response = requests.get('https://api.vk.com/method/groups.get', params=params)

            return response.json()

        except Exception as e:
            print(f'Ошибка выполнения запроса: {e}')
            return None

    def create_distr_groups(self, data):
        cities = []
        for group in data['response']['items']:
            if 'city' in group and group['city']:
                cities.append(group['city']['title'])

        return dict(Counter(cities))

    def normalize(self, counter_dict):
        total = sum(counter_dict.values())
        return {k: v / total for k, v in counter_dict.items()} if total > 0 else {}

    def extract_city(self):

        friends_distr = self.normalize(self.create_distr_friends(self.friends))
        groups_distr = self.normalize(self.create_distr_groups(self.groups))

        all_cities = set(friends_distr) | set(groups_distr)

        final_distr = {}
        for city in all_cities:
            friend_part = self.Wuser * friends_distr.get(city, 0)
            group_part = self.Wcomm * groups_distr.get(city, 0)
            final_distr[city] = friend_part + group_part

        final_distr = self.normalize(final_distr)

        return {
            'friends_distr': sorted(friends_distr.items(), key=lambda x: x[1], reverse=True),
            'groups_distr': sorted(groups_distr.items(), key=lambda x: x[1], reverse=True),
            'final_distr': sorted(final_distr.items(), key=lambda x: x[1], reverse=True),
            'city_pred': max(final_distr.items(), key=lambda x: x[1])[0] if final_distr else None
        }
load_dotenv()

cp = CityPredictor(os.getenv('TOKEN'), 166899514)

user_data = cp.extract_city()

print(f'Распределение по друзьям: {user_data['friends_distr']}')
print(f'Распределение по группам: {user_data['groups_distr']}')
print(f'Итоговое: {user_data['final_distr']}')
print(f'Город: {user_data['city_pred']}' )

