# Istruzioni

## Autori
* Gabriele Carrara: 814720
* Alberto Filosa: 815589
* Simone Tufano: 816984

## Librerie Utilizzate:

```py
import pandas as pd                  #-- Work with DataFrames
import numpy as np                   #-- Work with DataFrames
import string                        #-- Work with String
import re                            #-- Regular Expression
import pickle                        #-- Save Objects
from matplotlib import pyplot as plt #-- Matplotlib
import seaborn as sns                #-- Plots
from pprint import pprint            #-- Multiple Prints
from collections import Counter      #-- Count Words
from wordcloud import WordCloud      #-- WordCloud

#-- Nltk Package
import nltk
from nltk.corpus import stopwords            #-- Stopwords  
from nltk.stem import PorterStemmer          #-- Stemmization
from nltk.tokenize import WordPunctTokenizer #-- Tokenization
from nltk.stem import WordNetLemmatizer      #-- Lemmatization

#-- Sklearn Package
from sklearn.feature_extraction.text import CountVectorizer      #-- Bag of Words
from sklearn.feature_extraction.text import TfidfVectorizer      #-- Tf-Idf
from sklearn.decomposition import TruncatedSVD                   #-- SVD
from sklearn.svm import SVC                                      #-- SVM
from sklearn.model_selection import train_test_split             #-- Split Dataset
from sklearn.metrics import classification_report                #-- Model Summary 
from sklearn.ensemble import RandomForestClassifier              #-- Random Forest
from sklearn.cluster import KMeans                               #-- K-means Algorithm
from sklearn.metrics import confusion_matrix                     #-- Confusion Matrix
from sklearn.metrics.cluster import normalized_mutual_info_score #-- Mutual Information Score
from sklearn.cluster import AgglomerativeClustering              #-- Hierarchical
from sklearn.neighbors import KNeighborsClassifier               #-- knn

#-- Scipy Package
from scipy.sparse import random as sparse_random

#-- Gensim Package
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel

import warnings
warnings.filterwarnings("ignore", category = FutureWarning) 
```

## File

* Dataset: [Amazon Fine Food](https://www.kaggle.com/snap/amazon-fine-food-reviews)
* Script: TM_progetto.ipynb;
* Report: TM_report.pdf
* Slides: TM_slide.pptx