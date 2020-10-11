# import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')

import tweepy
import time
import datetime
import re
import json
import operator
import nltk
import os
import string
import copy
import _pickle as pickle
import math
import matplotlib.pyplot as plt
import re
import glob
import scipy.spatial
import json
import numpy as np

from flask import Flask
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
from pprint import pprint
from sklearn import preprocessing
from dateutil.parser import parse
from sentimentanalysis.sentiment import SentimentAnalysis
from flask import request, jsonify

model_path = './temp/covid19/model/covid19.pt'
vocab_path = './temp/covid19/data/demo.vocab.pickle'
config_path = './config.pickle'

from model import Cross_SEAN
cross_sean = Cross_SEAN(vocab_path, config_path, model_path)

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
sentiment_analysis = SentimentAnalysis(filename='sentimentanalysis/SentiWordNet.txt', weighting='harmonic')
stop_words = stopwords.words('english')
stop_words.append('rt')
stop_words.append('\n')

app = Flask(__name__)
# app.run(debug=True)
app.run(host= '0.0.0.0')
downloaded_tweets = "downloaded_tweets/"


def hydrate_tweet(tweet_id, save_dir):
    ##### Add your API credentials here  #####
    CONSUMER_KEY = ' '
    CONSUMER_SECRET = ' '
    OAUTH_TOKEN = ' '
    OAUTH_TOKEN_SECRET = ' '
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
    api = tweepy.API(auth)  

    try:
        tweet = api.get_status(tweet_id)._json
        with open(downloaded_tweets+save_dir+"/"+str(tweet_id) + '.json', 'w') as f:
            json.dump(tweet, f)
        return tweet
    except:
        return -2

def get_tweet_text(tweet_object):
    return tweet_object['text']

def get_processed_text(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-zA-Z@]', ' ', text).lower()
    tokens = [w for w in w_tokenizer.tokenize(text) if w not in stop_words and w[0] != '@']
    return ' '.join(tokens)

def tweet_extra_features(text):
    senti_score = sentiment_analysis.score(text)
    tokens = nltk.word_tokenize(text.lower())
    tags = nltk.pos_tag(nltk.Text(tokens))
    counter_ = Counter(tag for word, tag in tags)
    prp = int(counter_['PRP'])  # personal pronoun	I, he, she
    prp_ = int(counter_['PRP$'])  # possessive pronoun	my, his, hers
    pronouns = prp + prp_
    adjectives = int(counter_['JJ']) + int(counter_['JJR']) + int(counter_['JJS'])  # adjective
    nouns = int(counter_['NN']) + int(counter_['NNS']) + int(counter_['NNP']) + int(counter_['NNPS'])  # nouns
    vb = int(counter_['VB']) + int(counter_['VBD']) + int(counter_['VBG']) + int(counter_['VBN']) + int(
        counter_['VBP']) + int(counter_['VBZ'])  # verbs
    return [senti_score, nouns, adjectives, pronouns, vb]

def tweet_special_chars(text):
    num_special = 0
    for char in text:
        if not char.isalnum():
            num_special += 1

#     if num_special < 0:
#         print("ERROR: Okay something went really wrong here...")
#         num_special = 0

    return num_special

def tweet_all_features(tweet):
    num_user_mentions = len(tweet['entities']['user_mentions'])
    num_hashtags = len(tweet['entities']['hashtags'])
    num_urls = len(tweet['entities']['urls'])
    num_favs = tweet['favorite_count']
    num_rts = tweet['retweet_count']

    if 'media' in tweet['entities']:
        num_media = len(tweet['entities']['media'])
    else:
        num_media = 0

    is_reply = 0
    if tweet['in_reply_to_status_id']:
        is_reply = 1

    num_special_chars = tweet_special_chars(tweet)
    text = get_tweet_text(tweet)
    tweet_length = len(text)
    extra_features = tweet_extra_features(text)

    verified = tweet['user']['verified']
    followers_count = tweet['user']['followers_count']
    friends_count = tweet['user']['friends_count']
    favourites_count = tweet['user']['favourites_count']
    statuses_count = tweet['user']['statuses_count']
    return [num_user_mentions, num_hashtags, num_urls, num_favs, num_rts, num_media, is_reply,
            num_special_chars, tweet_length] + extra_features

def user_all_features(tweet):
    verified = tweet['user']['verified']
    followers_count = tweet['user']['followers_count']
    friends_count = tweet['user']['friends_count']
    favourites_count = tweet['user']['favourites_count']
    statuses_count = tweet['user']['statuses_count']

    return [verified, followers_count, friends_count, favourites_count, statuses_count]

def process(tweet_id, save_dir):
    tweet_object = hydrate_tweet(tweet_id, save_dir)

    if tweet_object == -2:
        return -2
    else:
        tweet_text = get_tweet_text(tweet_object)
        processed_text = get_processed_text(tweet_text)
        tweet_features = tweet_all_features(tweet_object) #
        user_features = user_all_features(tweet_object)

        return_data = cross_sean.predict(processed_text, tweet_features, user_features)
        return return_data

errors = {-1:"Not in status page!", 1:"Success", -2: "Tweet ID not found"}

@app.route('/predict_tweet')
def predict_tweet():
    save_dir = "all"
    
    tweet_id = request.args.get("data", "")
    print("check_tweet: tweet_id", tweet_id)
    
    return_data = process(tweet_id, save_dir)
    print(return_data)
    
    return return_data

@app.route('/report_fake')
def report_fake():
    save_dir = "fake"
    tweet_id = request.args.get("data", "")
    print("tweet_id", tweet_id)
    
    return_data = hydrate_tweet(tweet_id, save_dir)
    if return_data != -2:
        return {'response': True}
    return {'response': False}

@app.route('/report_genuine')
def report_genuine():
    save_dir = "genuine"
    tweet_id = request.args.get("data", "")
    print("tweet_id", tweet_id)
    
    return_data = hydrate_tweet(tweet_id, save_dir)
    if return_data != -2:
        return {'response': True}
    return {'response': False}

@app.route('/read_tweet')
def read_tweet():
    save_dir = "all"
    tweet_id = request.args.get("data", "")
    print("tweet_id", tweet_id)
    
    return_data = hydrate_tweet(tweet_id, save_dir)
    if return_data != -2:
        tweet_id = return_data['id']
        tweet_text = get_tweet_text(return_data)
        tweet_sentiment = sentiment_analysis.score(tweet_text)
        return {'tweet_id': tweet_id, 'tweet_text': tweet_text, 'tweet_sentiment': tweet_sentiment}
    return False