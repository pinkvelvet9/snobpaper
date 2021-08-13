# This code computes the chi-squared value for each word in the vocabulary relative to each occupation to obtain a vocabulary for the task-irrelevant gender classifiers. Run it as `python3 get_chi.py`.

from collections import Counter
import http.client, urllib.parse, json, time, sys
from glob import glob
from bs4 import BeautifulSoup as Soup
sys.path.append("../words")
import we
from sklearn.svm import LinearSVC, SVC
import numpy as np
import re, sys
import random
import scipy
from adjustText import adjust_text

import statsmodels.stats.proportion
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import sklearn.feature_selection 
from nltk.stem.porter import *
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import time
import os, json
import re
import textblob
import langdetect
import pickle as pkl
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from bs4 import BeautifulSoup
from random import randint
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

bios = pkl.load(open('biosbias/data/BIOS_inferred.pkl','rb'))

data = bios
print(f'Data: {len(data)}')

# stratified split
labels_all = [bio['title'] for bio in data]
bios_data_train_val, bios_data_test = train_test_split(data, test_size=0.20, random_state=42, stratify=labels_all)

labels_train_val = [bio['title'] for bio in bios_data_train_val]
bios_data_train, bios_data_val = train_test_split(bios_data_train_val, test_size=0.25, random_state=42, stratify=labels_train_val)

vectorizer = CountVectorizer(analyzer='word',min_df=0.001,binary=False)
X_train = vectorizer.fit_transform([p["bio"] for p in bios_data_train])
words = np.array(vectorizer.get_feature_names())
occs = ["surgeon", "software_engineer", "composer", "pastor", "comedian", "architect", "chiropractor", "accountant", "attorney", "filmmaker", "physician", "dentist", "photographer", "professor", "painter", "journalist", "poet", "personal_trainer", "teacher", "psychologist",  "model", "interior_designer", "yoga_teacher", "nurse", "dietitian"]
#

f_chi={}

for occ in occs:
    print(occ)
    is_occ = [p['title'] ==occ for p in bios_data_train if p['gender']=='F']
    words_chi = [sklearn.feature_selection.chi2(np.array([p['bio'].count(w) for p in bios_data_train if p['gender']=='F']).reshape(-1, 1),is_occ) for w in words]
    f_chi[occ] = words_chi

pkl.dump(f_chi,open('results/f_chi.pkl','wb'),-1)


m_chi={}
for occ in occs:
    print(occ)
    is_occ = [p['title'] ==occ for p in bios_data_train if p['gender']=='M']
    words_chi = [sklearn.feature_selection.chi2(np.array([p['bio'].count(w) for p in bios_data_train if p['gender']=='M']).reshape(-1, 1),is_occ) for w in words]
    m_chi[occ] = words_chi

pkl.dump(m_chi,open('results/m_chi.pkl','wb'),-1)
