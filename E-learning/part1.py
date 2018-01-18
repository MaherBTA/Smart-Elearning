
# coding: utf-8

# In[1]:


import numpy as np
import scipy
import io
import nltk
import pandas as pd
import os,fnmatch
import re
import matplotlib as plt
import itertools
import codecs
import tensorflow as tf


# In[2]:


from sklearn.neural_network import MLPClassifier
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn import svm 
from sklearn.naive_bayes import MultinomialNB

from tensorflow.contrib import learn
from nltk.corpus import stopwords


# In[3]:


from scipy.sparse import hstack


# In[4]:


# Download nltk.corpus firstly , 
nltk.download(info_or_id='punkt', download_dir='/home/sneaky/nltk_data')
nltk.download(info_or_id='averaged_perceptron_tagger', download_dir='/home/sneaky/nltk_data')
nltk.download(info_or_id='stopwords', download_dir='/home/sneaky/nltk_data')
nltk.download(info_or_id='wordnet', download_dir='/home/sneaky/nltk_data')


# In[5]:


from collections import Counter
import string
import math
from nltk import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 


# In[6]:

# In[7]:


def remove_stop(sentances,noise_list):
    sent_list = []
    for sent in sentances:
        words = sent.split() 
        noise_free_words = [word for word in words if word not in noise_list] 
        noise_free_text = " ".join(noise_free_words) 
  #      print(noise_free_text)
        sent_list = sent_list + [noise_free_text]
    return sent_list


def stem_sent(sentances):
    sent_list = []
    lem = WordNetLemmatizer()
#    stem = PorterStemmer()
    for sent in sentances:
        words = sent.split()       
        words_stemed = [lem.lemmatize(word, "v") for word in words]
        stemmed_text  = " ".join(words_stemed)
        sent_list = sent_list + [stemmed_text]
    return sent_list


# In[8]:


def cleaning_text(text,noise_list):
    text = text.replace('..', '.')
#   text = text.replace('?', '')
    sentances = sent_tokenize(text)
    # Remove Punctionation
 #   exclude = set(string.punctuation)
 #   sentances = [''.join(ch for ch in sent if ch not in exclude) for sent in sentances]

#-------------- Noise removal -----------------------------------###
    sentances = remove_stop(sentances,noise_list)
    sentances = stem_sent(sentances)
    return sentances


def load_data_to_corpus(df_classes,df_courses):
# Import class list from .txt file 
# Import Corpus of texts
# Import Corpus of QCM
       
    ### ----- Import class names from Excel file -----###
    ls_classes = list(df_classes['class_name']) 
    lst_text = list(df_classes['text_file'])
    lst_qcm = list(df_classes['qcm_file'])
    lst_courses = list(df_classes['class_id'])
    
    ### --------------------------------------------------
    
    ###--------- Import the list of stopwords ------###
    Stop =  stopwords.words('english')
    ### ------------ Load Texts --------------------###
    path = path = os.getcwd() + "/" + "textes"    
    train_corpus = get_token_by_folder(path,Stop,ls_classes,lst = lst_text)
    ### ------------ Load QCM Texts ---------------- ###
    ### Load from .txt files
    path1 = os.getcwd() + "/" + "qcm"
#    test_corpus = get_token_by_folder(path1,Stop,ls_classes,lst = lst_qcm)
    ### Load from .xlsx file
    test_corpus = get_token_by_file(df_courses,Stop,ls_classes,lst_courses)
    # ---------------- End ---------------------------##
    ####################################################
    return [train_corpus,test_corpus]


# In[15]:


def corpus_to_train(train_corpus,test_corpus):
    train_data = []
    train_labels = []
    print('train_corpus=',train_corpus)
    for row in train_corpus:
        train_data.append(row[0])
        train_labels.append(row[1])
    test_data = []
    test_labels = [] 
    for row in test_corpus:
        test_data.append(row[0]) 
        test_labels.append(row[1])
    return [train_data,train_labels,test_data,test_labels]


# # Main program of first part

# In[16]:


### ---------- Main program --------- ##
[df_classes,df_courses] = getDfbyfolder("",'train')
[train_corpus,test_corpus] = load_data_to_corpus(df_classes,df_courses)
[train_data,train_labels,test_data,test_labels] = corpus_to_train(train_corpus,test_corpus)


# In[21]:


# Create feature vectors 
### ---- Put here all features 
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
count_vect = CountVectorizer()



list_comb = [('tf', vectorizer),('cnt',count_vect)]
combined_f = FeatureUnion(list_comb)

# -----------

prediction = []

X_train2 = combined_f.fit_transform(train_data)


# ---------------
model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train2, train_labels)
# -----
def predict_text(model,input_text):
    X_test2 = combined_f.transform([input_text])
    prediction = model.predict(X_test2)
    return prediction


p = predict_text(model_svm,train_data[0])
print(p)


