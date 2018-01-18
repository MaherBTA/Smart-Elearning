
# coding: utf-8

# In[59]:


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


# In[60]:


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


# In[61]:


from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest


# In[62]:


from scipy.sparse import hstack


# In[63]:

nltk.data.path.append(os.getcwd()+"/nltk_data");

# Download nltk.corpus firstly , 
nltk.download(info_or_id='punkt')
nltk.download(info_or_id='averaged_perceptron_tagger')
nltk.download(info_or_id='stopwords')
nltk.download(info_or_id='wordnet')


# In[64]:


from collections import Counter
import string
import math
from nltk import word_tokenize,sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer 


# In[65]:


def get_folder_content(path,filtere):
    lst = sorted(fnmatch.filter(os.listdir(path), filtere))
    return lst


# In[66]:


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


# In[67]:


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


# In[68]:


def get_token_by_folder(path,noise_list,ls_classes,lst):
    corpus = []
    if lst is None : # Get all the folder content
        lst = get_folder_content(path,'*.txt')
 
    for i in range(len(lst)):
        with codecs.open(path+'/'+ lst[i],encoding='utf-8') as shakes:
            text = shakes.read()
        # If the sentance contains two .. , we must suppress one to improve the spliting output
        sentances = cleaning_text(text,noise_list)

##### --------- Add the class label to sent ---------------------###
        corpus = corpus + [(sent,ls_classes[i]) for sent in sentances]

    return corpus


# In[69]:


def get_qcm_by_course(df1,course_id):
    df2 = df1[df1['Course'] == course_id]
    ch = ""
    if(list(df2.iterrows()) != []):
        for index, row in df2.iterrows():
            if (row['Type'] == 'tf'):
                ch1 = "\n"+ row['Question_text']
            else : 
                if(str(row['Responses_list']) not in ["nan"," "]):
                    ch1 =  "\n"+ row['Question_text']+"\n"+ row['Responses_list']+"\n"
                    ch = ch + ch1
    return ch


# In[70]:


def get_token_by_file(df,noise_list,ls_classes,lst_course):
    corpus = []
    
#    lst_course = ['CH1C1','CH1C2','CH1C3','CH1C4','CH1C5','CH1C6','CH1C7']
    for i in range(len(lst_course)):
        course_id = lst_course[i]
        text = get_qcm_by_course(df,course_id)
        if (text != ""):
#-------------- Noise removal -----------------------------------###
            sentances = cleaning_text(text,noise_list)

##### --------- Add the class label to sent ---------------------###
            corpus = corpus + [(sent,ls_classes[i]) for sent in sentances]

    return corpus


# In[71]:


def Exelfile_to_df(file_name):   
    xl_file = pd.ExcelFile(file_name)
    dfs = {sheet_name: xl_file.parse(sheet_name) 
          for sheet_name in xl_file.sheet_names}
    return dfs
def get_sheet_names(dfs):
    return list(dfs)   
def get_df_by_sheet(dfs,sheet_name):
    df = dfs[sheet_name]
    return df


# In[72]:


def getDfbyfolder(folder_name,task):
    if (task == 'train'):
        fname = "data_train_qcm.xlsx"
    elif (task == 'test'):
        fname = "data_test_qcm.xlsx"
    
    if folder_name == "":
        file_name = os.getcwd() + "/" + "qcm" + "/" + fname
    else : 
        file_name = os.getcwd() + "/" + folder_name + "/" + "qcm" + "/" + fname
    dfs = Exelfile_to_df(file_name)  ## Import the Excel file of classes
    ls_sheets = get_sheet_names(dfs) ## 
    frames = []
    if (ls_sheets[-1] == "classes"):
        for sh in ls_sheets[:-1]:
            df = dfs[sh]
            df = df[df.Question_text.notnull()]   
            frames.append(df)
        df_classes = get_df_by_sheet(dfs,"classes")
    else:
        for sh in ls_sheets:
            df = dfs[sh]
            df = df[df.Question_text.notnull()]   
            frames.append(df)
        df_classes = None   
    df_courses = pd.concat(frames,ignore_index = True)#get_df_by_sheet(dfs,ls_sheets[0])  # Get Only the first Sheet
    return [df_classes,df_courses]


# In[73]:


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


# In[74]:


def corpus_to_train(train_corpus,test_corpus):
    train_data = []
    train_labels = []
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

# In[75]:


### ---------- Main program --------- ##
[df_classes,df_courses] = getDfbyfolder("",'train')
[train_corpus,test_corpus] = load_data_to_corpus(df_classes,df_courses)
[train_data,train_labels,test_data,test_labels] = corpus_to_train(train_corpus,test_corpus)


# In[56]:


######## Test Combined features and All Features Caract 


# In[57]:


# Create feature vectors 
### ---- Put here all features 
dict_vect = HashingVectorizer()
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
count_vect = CountVectorizer()

# This dataset is way too high-dimensional. Better do PCA:
pca = PCA(n_components=2)
# Maybe some original features where good, too?
selection = SelectKBest(k=1)
# Build estimator from PCA and Univariate selection:


#list_comb = [('tf', vectorizer), ('cnt',count_vect), ('hs',dict_vect) ]
list_comb = [('tf', vectorizer),('cnt',count_vect)]
combined_f = FeatureUnion(list_comb)
# Train the feature vectors
#X_train1 = count_vect.fit_transform(train_data)
#X_test1 = count_vect.transform(test_data)

# -----------

prediction = []

X_train2 = combined_f.fit_transform(train_data)
X_test2 = combined_f.transform(test_data)

# ---------------
model_svm = svm.SVC(kernel='linear') 
model_svm.fit(X_train2, train_labels) 
# -----
prediction = model_svm.predict(X_test2)

#print (classification_report(test_labels, prediction))


# # Second Part 

# In[58]:


def load_data_to_df(features_list = ['Question_text','Responses_list'],labels_list = ['level']):
    F1= features_list[0]
    F2 = features_list[1]
    L1 = labels_list[0]
    #### ------- Load data from Excel file ---------#####
    
    #### --------- Get Train Data ------------------#####

    [df_classes,df_courses] = getDfbyfolder("",'train')
    ####---------
    qdf_train = list(df_courses[F1])  # Feature 1

    rdf_train = list(df_courses[F2]) # Feature 2

    
    ### ---------
    train_labels = list(df_courses[L1])

    #### --------- Get Test Data ------------------#####
    [df_N,df_test] = getDfbyfolder("",'test')
    ####---------
    qdf_test = list(df_test[F1])  # Feature 1
    rdf_test = list(df_test[F2]) # Feature 2

    ### ---------
    test_labels = list(df_test[L1])

    train_data = [qdf_train,rdf_train]
    test_data = [qdf_test,rdf_test]
    return [train_data,train_labels,test_data,test_labels]
    


# In[29]:


[train_data,train_labels,test_data,test_labels] = load_data_to_df(features_list = ['Question_text','Responses_list'],labels_list = ['level'])
# Create feature vectors 
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
count_vect = CountVectorizer()#TfidfTransformer()#

X_train1 = count_vect.fit_transform(train_data[0])
X_test1 = count_vect.transform(test_data[0])

X_train2 = count_vect.fit_transform(train_data[1])
X_test2 = count_vect.transform(test_data[1])


X_train = hstack((X_train1, X_train2))
X_test = hstack((X_test1, X_test2))

prediction = []

model_svm = svm.SVC(kernel='linear')
model_svm.fit(X_train, train_labels)

print(X_test)

#print(X_test2.shape[1])
prediction = model_svm.predict(X_test)
#print (classification_report(test_labels, prediction))

