# This code trains occupation classifiers based on word-embedding representations of the bios, using the Fairlearn ExponentiatedGradient method to increase True Positive Rate parity. However, due to the sparse representation, it is not effective at doing so. 

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

from fairlearn.reductions import ExponentiatedGradient, EqualizedOdds
from sklearn.linear_model import LogisticRegression

from collections import Counter
import http.client, urllib.parse, json, time, sys
from glob import glob
sys.path.append("../words")
# import we
from sklearn.svm import LinearSVC, SVC
import numpy as np
import re, sys
import random
import scipy
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import sklearn.feature_selection 
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import time
import os, json
import re
import pickle as pkl
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold
from random import randint
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
import copy
np.random.seed(0)  

from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.linear_model import LogisticRegression

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
import we
import os

import numpy as np
# from mag.experiment import Experiment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# bios = pkl.load(open('biosbias/data/BIOS_inferred.pkl','rb'))
# data = np.array(bios)
# print(f'Data: {len(data)}')

bios = pkl.load(open('fairbios/data/BIOS_inferred.pkl','rb'))
# def load_data(filename):
data = bios
print(f'Data: {len(data)}')

# stratified split
labels_all = [bio['title'] for bio in data]
bios_data_train_val, bios_data_test = train_test_split(data, test_size=0.20, random_state=42, stratify=labels_all)

labels_train_val = [bio['title'] for bio in bios_data_train_val]
bios_data_train, bios_data_val = train_test_split(bios_data_train_val, test_size=0.25, random_state=42, stratify=labels_train_val)

eo_y_preds = {}

# fold_ind = 0
# for train_index, test_index in skf.split(data, labels_all):
#     bios_data_train, bios_data_test = data[train_index], data[test_index]


vectorizer = CountVectorizer(analyzer='word',min_df=0.001,binary=False)
X_train = vectorizer.fit_transform([p["bio"] for p in bios_data_train])
X_test = vectorizer.transform([p["bio"] for p in bios_data_test])
print("Done featurizing")

G_train = [p["gender"] for p in bios_data_train]
G_test = [p["gender"] for p in bios_data_test]


np.random.seed(0)  
occs = ["surgeon", "software_engineer", "composer", "pastor", "comedian", "architect", "chiropractor", "accountant", "attorney", "filmmaker", "physician", "dentist", "photographer", "professor", "painter", "journalist", "poet", "personal_trainer", "teacher", "psychologist",  "model", "interior_designer", "yoga_teacher", "nurse", "dietitian"]
eo_mitigators_train = {}
 
G_train = [p["gender"] for p in bios_data_train]
for target_occ in occs:
    print(target_occ)
    Y_train = [p["title"]==target_occ for p in bios_data_train]
    constraint = TruePositiveRateParity()
    classifier = sklearn.linear_model.SGDClassifier(loss='log',class_weight='balanced')
    mitigator = ExponentiatedGradient(classifier, constraint)
    mitigator.fit(np.array(X_train), np.array(Y_train), sensitive_features=np.array(G_train))
    eo_mitigators_train[target_occ] = mitigator
    y_pred_mitigated = eo_mitigators_train[target_occ]._pmf_predict(X_test)

    eo_y_preds[target_occ] = y_pred_mitigated
    pkl.dump(eo_y_preds,open("results/bow_tpr_sgd_preds.pkl","wb"))
