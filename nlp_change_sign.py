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
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import roc_curve, auc

import warnings
warnings.filterwarnings('ignore')

USE_RES_SIGN=False

if USE_RES_SIGN is True:
    TARGET='res_sign'
else:
    TARGET='Change_sign'


tweets_tesla=pd.read_csv('tweets_tesla.csv', index_col='Date')

dataset=tweets_tesla[['tweet_hashtags',TARGET]]

dataset.rename(columns={'tweet_hashtags': 'text',TARGET:'target'}, inplace=True)



dataset['text']=dataset['text'].str.lower()


# Import the NLTK package and download the necessary data
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# view the stopwords
ENGstopwords = stopwords.words('english')
#print(ENGstopwords[0:10])


STOPWORDS = set(ENGstopwords)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])
dataset['text'] = dataset['text'].apply(lambda text: cleaning_stopwords(text))



dataset['text']=dataset['text'].apply(lambda x: cleaning_stopwords(x))


import string
english_punctuations = string.punctuation
punctuations_list = english_punctuations
def cleaning_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)

dataset['text']= dataset['text'].apply(lambda x: cleaning_punctuations(x))

import re
def cleaning_repeating_char(text):
    return re.sub(r'(.)1+', r'1', text)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_repeating_char(x))

#Cleaning and removing numeric numbers

def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)
dataset['text'] = dataset['text'].apply(lambda x: cleaning_numbers(x))


#Getting tokenization of tweet text

def tokenize(text):
    tokens = text.split()
    return tokens

dataset['text'] = dataset['text'].apply(lambda x: tokenize(x))

#Applying stemming

import nltk
st = nltk.PorterStemmer()
def stemming_on_text(x):
    y = [st.stem(word) for word in x]
    return y

dataset['text']= dataset['text'].apply(lambda x: stemming_on_text(x))


#Applying lemmatizer

lm = nltk.WordNetLemmatizer()
def lemmatizer_on_text(data):
    y = [lm.lemmatize(word) for word in data]
    return y

dataset['text'] = dataset['text'].apply(lambda x: lemmatizer_on_text(x))



#--------------------
data=tweets_tesla[['tweet_hashtags',TARGET]]

data.rename(columns={'tweet_hashtags': 'text',TARGET:'target'}, inplace=True)

#Separating input feature and label
X=data.text
y=data.target

#Splitting Our Data Into Train and Test Subsets
# Separating the 80% data for training data and 20% for testing data
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state =26105111)

X_train=X[:-15]
X_test=X[-15:]

y_train=y[:-15]
y_test=y[-15:]

#Transforming the Dataset Using TF-IDF Vectorizer
# Fit the TF-IDF Vectorizer

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500)
vectoriser.fit(X_train)
print('No. of feature_words: ', len(vectoriser.get_feature_names()))


#Transform the data using TF-IDF Vectorizer

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)


#Function for Model Evaluation
#After training the model, we then apply the evaluation measures to check how the model is performing. Accordingly, we use the following evaluation parameters to check the performance of the models respectively:

#Accuracy Score
#Confusion Matrix with Plot
#ROC-AUC Curve
def model_Evaluate(model):
# Predict values for Test dataset
    y_pred = model.predict(X_test)
    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    labels = [f'{v1}n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
    xticklabels = categories, yticklabels = categories)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
    
    
    
#Model Building
#Model-1
print('------------Model BernoulliNB--------')
BNBmodel = BernoulliNB()
BNBmodel.fit(X_train, y_train)
model_Evaluate(BNBmodel)
y_pred1 = BNBmodel.predict(X_test)
y_pred1_train=BNBmodel.predict(X_train)

predict_tweet=np.concatenate((y_pred1, y_pred1_train), axis=0)
print(len(predict_tweet))                    

print(len(tweets_tesla))

tweets_tesla['predict_tweet']=predict_tweet
print(tweets_tesla.head())
tweets_tesla.to_csv('tweets_predictions_tesla.csv')



y_prob1=BNBmodel.predict_proba(X_train)
y_prob1=y_prob1[:,1]

y_prob1_1=BNBmodel.predict_proba(X_test)
y_prob1_1=y_prob1_1[:,1]



fpr, tpr, thresholds = roc_curve(y_train, y_prob1)
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob1_1)


roc_auc = auc(fpr, tpr)
roc_auc1 = auc(fpr1, tpr1)

plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label='ROC curve Train(area = %0.2f)' % roc_auc)
plt.plot(fpr1, tpr1, color='orange', lw=1, label='ROC curve Test(area = %0.2f)' % roc_auc1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE PROB-BernoulliNB')
plt.legend(loc="lower right")
plt.savefig('./data/ROC CURVE.png')

plt.show()


#Model-2:
print('------------Model SVCmodel--------')


SVCmodel = LinearSVC(class_weight='balanced')
SVCmodel.fit(X_train, y_train)
model_Evaluate(SVCmodel)
y_pred2 = SVCmodel.predict(X_test)



#Model-3
print('------------Model LogisticRegression--------')


LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1, class_weight='balanced')
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel)
y_pred3 = LRmodel.predict(X_test)

y_prob3=LRmodel.predict_proba(X_train)
y_prob3=y_prob3[:,1]

y_prob3_1=LRmodel.predict_proba(X_test)
y_prob3_1=y_prob3_1[:,1]



fpr, tpr, thresholds = roc_curve(y_train, y_prob3)
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob3_1)


roc_auc = auc(fpr, tpr)
roc_auc1 = auc(fpr1, tpr1)

plt.figure()
plt.plot(fpr, tpr, color='green', lw=1, label='ROC curve Train(area = %0.2f)' % roc_auc)
plt.plot(fpr1, tpr1, color='orange', lw=1, label='ROC curve Test(area = %0.2f)' % roc_auc1)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC CURVE PROB')
plt.legend(loc="lower right")
plt.show()



#Model-4


#Model-3
print('------------Model RandomForestClassifier--------')

def new_model(model, X_train, y_train, X_test, y_test):   
    
    model.fit(X_train, y_train)
    model_Evaluate(model)
    y_pred3 = model.predict(X_test)
    
    y_prob3=model.predict_proba(X_train)
    y_prob3=y_prob3[:,1]
    
    y_prob3_1=model.predict_proba(X_test)
    y_prob3_1=y_prob3_1[:,1]
    
    
    
    fpr, tpr, thresholds = roc_curve(y_train, y_prob3)
    fpr1, tpr1, thresholds1 = roc_curve(y_test, y_prob3_1)
    
    
    roc_auc = auc(fpr, tpr)
    roc_auc1 = auc(fpr1, tpr1)
    
    plt.figure()
    plt.plot(fpr, tpr, color='green', lw=1, label='ROC curve Train(area = %0.2f)' % roc_auc)
    plt.plot(fpr1, tpr1, color='orange', lw=1, label='ROC curve Test(area = %0.2f)' % roc_auc1)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC CURVE PROB')
    plt.legend(loc="lower right")
    plt.show()
    

new_model(RandomForestClassifier(), X_train, y_train, X_test, y_test)
  




















#print(data.tail())

