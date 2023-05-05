import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from datetime import timedelta


# sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


tweets_tesla=pd.read_csv('tweets_tesla.csv', index_col='Date')

dataset=tweets_tesla[['tweet_hashtags','Change_sign']]

dataset.rename(columns={'tweet_hashtags': 'text','Change_sign':'target'}, inplace=True)