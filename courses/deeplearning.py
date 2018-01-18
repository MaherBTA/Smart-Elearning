from .models import *
import numpy as np
import scipy
import io
import nltk
import pandas as pd
import os,fnmatch
import re
import itertools
import codecs
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

from scipy.sparse import hstack

nltk.data.path.append(os.getcwd()+"/nltk_data");

# Download nltk.corpus firstly , 
nltk.download(info_or_id='punkt')
nltk.download(info_or_id='averaged_perceptron_tagger')
nltk.download(info_or_id='stopwords')
nltk.download(info_or_id='wordnet')



from collections import Counter
import string
import math
from nltk import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 


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



def cleaning_text(questions,noise_list):
    result=[]
    for text in questions:
        text = text.content.replace('..', '.')
    #   text = text.replace('?', '')
        sentances = sent_tokenize(text)
        # Remove Punctionation
     #   exclude = set(string.punctuation)
     #   sentances = [''.join(ch for ch in sent if ch not in exclude) for sent in sentances]
    
    #-------------- Noise removal -----------------------------------###
        sentances = remove_stop(sentances,noise_list)
        sentances = stem_sent(sentances)
        result=result + sentances
    return result


# # Main program of first part
def get_relevent_questions(input_text):
    
    questions = Question.objects.all()
    train_data=[]
    train_labels=[]
    for question in questions :
        train_labels=train_labels + [question.id]
        answers = Answers.objects.filter(question=question.id)
        text=question.content
        for obj in answers: 
            text =  text +" "+ obj.answer
            train_data=train_data+[text]
        
    Stop = stopwords.words('english')
    train_data = cleaning_text(questions,Stop)
    
    # Create feature vectors 
    ### ---- Put here all features 
    dict_vect = HashingVectorizer()
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
    count_vect = CountVectorizer()
    
    #list_comb = [('tf', vectorizer), ('cnt',count_vect), ('hs',dict_vect) ]
    list_comb = [('tf', vectorizer),('cnt',count_vect)]
    combined_f = FeatureUnion(list_comb)
    
    X_train2 = combined_f.fit_transform(train_data)
    
    # ---------------
    model_svm = svm.SVC(kernel='linear',probability=True) 
    model_svm.fit(X_train2, train_labels)
    # -----
    X_test2 = combined_f.transform([input_text])
    proba = model_svm.predict_proba(X_test2)
    predict_proba = np.array(model_svm.classes_)
    proba = np.array(proba)
    inds = proba.argsort()
    predict_proba = predict_proba[inds].tolist()
    #print (classification_report(test_labels, prediction))
    # In[24]:
    return predict_proba[0][:6]

def reduce_dementions(MD_train_data):
    train_data=[]
    for steps in MD_train_data:
        list=[0,0,0,0,0]
        for step in steps:
            list[0] = list[0] + step.s_type
            list[1] = list[1] + step.s_level
            list[2] = list[2] + step.q_type
            list[3] = list[3] + step.q_level
            list[4] = list[4] + step.is_correct
            #list[5] = list[5] + step.question
        list = [x / len(MD_train_data) for x in list]
        #list[5]=hash(list[5])
        train_data.append(list)
    return train_data

# # Main program of first part
def get_next_question(progress):
    MD_train_data=[]
    train_labels=[]
    progresses = Progress.objects.all()
    for item in progresses:
        train_labels.append(item.predicted_question)
        steps = Steps.objects.filter(progress=item)
        MD_train_data.append(steps)
    train_data=reduce_dementions(MD_train_data)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_data, train_labels)
    return clf.predict([progress])

