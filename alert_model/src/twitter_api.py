"""Module used to retrieve Twitter data_folder."""

import time
import pandas as pd
import tweepy
import os.path
from tqdm import tqdm
from startup import TWITTER_DATA, PROCESSED_DATA

# credentials of academic Twitter developer API
consumer_key = "xxxxxxxxxxx"
consumer_secret = "xxxxxxxx"
access_token = "xxxxxxxx"
access_token_secret = "xxxxxxxx"
API_key = "xxxxxxxx"
API_key_secret = "xxxxxxxxxx"
bearer_token = "xxxxxxxxxx"

client = tweepy.Client(bearer_token=bearer_token,
                     consumer_key=consumer_key,
                     consumer_secret=consumer_secret,
                     access_token=access_token,
                     access_token_secret=access_token_secret,
                     wait_on_rate_limit=True)

def get_tweet_count(symptom: str, date_start: str, date_end: str):
    """
    get the count of tweet per day.
    """
    file_path = f'{TWITTER_DATA}/{symptom}_tweets_count.csv'
    if not os.path.exists(file_path):
        queryTopic = f'{symptom} lang:de'
        count_list = []
        for i in tweepy.Paginator(client.get_all_tweets_count, query=queryTopic, end_time=date_end, start_time=date_start, granularity='day').flatten(limit=10000000):
            count_list.append({'date': i['start'][:10], 'tweet_count': i['tweet_count']})
            time.sleep(0.1)

        df = pd.DataFrame(count_list, columns=['date', 'tweet_count'])
        df.to_csv(file_path, encoding='utf_8_sig')

def read_tweet_count(syn_list: list):
    """
    Read all symptom related tweets count files and combine into a new dataframe.
    """
    new_df = pd.DataFrame()
    for sym in tqdm(syn_list, desc="Iterating through symptoms"):
        df = pd.read_csv(f'{TWITTER_DATA}/{sym}_tweets_count.csv')
        if not (df['tweet_count']==0).all():
            new_df['date'] = df['date']
            new_df[sym] = df['tweet_count']
    new_df = new_df.sort_values(['date'])
    new_df.set_index(new_df['date'])
    print(new_df.shape)
    print(new_df.columns)
    new_df.to_csv(f'{PROCESSED_DATA}/daily_twitter_german.csv', encoding='utf_8_sig')


if __name__ == "__main__":
