import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.stem.snowball import SnowballStemmer
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers   import KerasRegressor
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.optim as optim

stemmer = SnowballStemmer('english')
stemming = input("Do you want to use stemming? (y/n): ")

df_train = pd.read_csv('train.csv/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('product_descriptions.csv/product_descriptions.csv')
num_train = df_train.shape[0]

def str_stemmer(s):
    if not isinstance(s, str):
        s = str(s)
    return " ".join([stemmer.stem(word) for word in s.lower().split()])

def str_common_word(str1, str2):
    return sum(int(str2.find(word)>=0) for word in str1.split())

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

df_all = df_all.rename(columns={'product_uid ': 'product_uid'})
df_all = pd.merge(df_all, df_pro_desc, how='left', on="product_uid")

if stemming == 'y':
    print("Executing with stemming")
    df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))
    df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))
    df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))
else:
    print("Executing without stemming")

df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))
df_all['new_feature'] = df_all['len_of_query'] * df_all['word_in_description']
df_all['new_feature2'] = df_all['len_of_query'] * df_all['word_in_title']
df_all['new_feature3'] = df_all['word_in_description'] * df_all['word_in_title']
df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1)
df_train = df_all.iloc[:num_train]

X = df_train.drop(['id', 'relevance'], axis=1)
y = df_train['relevance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

