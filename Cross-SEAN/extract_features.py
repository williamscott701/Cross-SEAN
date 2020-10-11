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
import sklearn

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

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
sentiment_analysis = SentimentAnalysis(filename='sentimentanalysis/SentiWordNet.txt', weighting='harmonic')
stop_words = stopwords.words('english')
stop_words.append('rt')
stop_words.append('\n')

def get_tweet_text(tweet_object):
    try:
        return tweet_object['full_text']
    except:
        return tweet_object['text']

def get_processed_text(pre):
    text = re.sub(r'http\S+', ' ', pre)
    text = re.sub(r'[^a-zA-Z@]', ' ', text).lower()
    tokens = [w for w in w_tokenizer.tokenize(text) if w[0] != '@' and w != 'rt' and w != '\n']
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

    if num_special < 0:
        print("ERROR: Okay something went really wrong here...")
        num_special = 0

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

    return [num_user_mentions, num_hashtags, num_urls, num_favs, num_rts, num_media, is_reply,
            num_special_chars, tweet_length] + extra_features

def user_all_features(tweet):
    verified = tweet['user']['verified']
    followers_count = tweet['user']['followers_count']
    friends_count = tweet['user']['friends_count']
    favourites_count = tweet['user']['favourites_count']
    statuses_count = tweet['user']['statuses_count']

    return [verified, followers_count, friends_count, favourites_count, statuses_count]

def process(tweet_object):
    tweet_text = get_tweet_text(tweet_object)
    processed_text = get_processed_text(tweet_text)
    tweet_features = tweet_all_features(tweet_object) #
    user_features = user_all_features(tweet_object)

    return_data = [tweet_text, processed_text, tweet_features, user_features]
    
    return return_data

split = 0.85

fake_text = []
fake_tweet_objs = []
fake_user_objs = []

with open('./data/covid19/fake.json') as f:
    ts = json.load(f)
    for t in ts:
        return_data = process(t)
        fake_text.append('1\t  'return_data[1])
        fake_tweet_objs.append(return_data[-2])
        fake_user_objs.append(return_data[-1])

genuine_text = []
genuine_tweet_objs = []
genuine_user_objs = []

with open('./data/covid19/genuine.json') as f:
    ts = genuine.load(f)
    for t in ts:
        return_data = process(t)
        genuine_text.append('0\t  ' + return_data[1])
        genuine_tweet_objs.append(return_data[-2])
        genuine_user_objs.append(return_data[-1])

        
all_text = fake_text + genuine_text
all_tfeats = fake_tweet_objs + genuine_tweet_objs
all_ufeats = fake_user_objs + genuine_user_objs

all_text, all_tfeats, all_ufeats = sklearn.utils.shuffle(all_text, all_tfeats, all_ufeats)

train_size = len(all_text)*split

train_text = all_text[:train_size]
test_text = all_text[train_size:]

train_tfeats = all_tfeats[:train_size]
test_tfeats = all_tfeats[train_size:]

train_ufeats = all_ufeats[:train_size]
test_ufeats = all_ufeats[train_size:]

with open('./data/covid19/train.txt', 'w') as f:
    f.write('\n'.join(train_text))
with open('./data/covid19/test.txt', 'w') as f:
    f.write('\n'.join(test_text))
    
np.save('./data/covid19/train/tweet_features_train.npy', train_tfeats)
np.save('./data/covid19/train/user_features_train.npy', train_ufeats)
np.save('./data/covid19/test/tweet_features_test.npy', test_tfeats)
np.save('./data/covid19/test/user_features_test.npy', test_ufeats)