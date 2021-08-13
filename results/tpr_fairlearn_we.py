# This code trains occupation classifiers based on word-embedding representations of the bios, using the Fairlearn ExponentiatedGradient method to increase True Positive Rate parity.

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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA


bios = pkl.load(open('biosbias/data/BIOS_inferred.pkl','rb'))
data = bios
print(f'Data: {len(data)}')

# stratified split
labels_all = [bio['title'] for bio in data]
bios_data_train_val, bios_data_test = train_test_split(data, test_size=0.20, random_state=42, stratify=labels_all)

labels_train_val = [bio['title'] for bio in bios_data_train_val]
bios_data_train, bios_data_val = train_test_split(bios_data_train_val, test_size=0.25, random_state=42, stratify=labels_train_val)

with open("../../words/embeddings/OtherFormats/crawl-300d-2M.pkl", "rb") as f:
    E = pkl.load(f)
print("Loaded", len(E), "words")

def sim(w1, w2):
    return E[w1].dot(E[w2])/linalg.norm(E[w1])/linalg.norm(E[w2])
 
# sim('he', 'she')
def word_vector_featurize(text, Emb = E):
    return np.mean([Emb[w] for w in re.split(r"[\s\.\!\?\:,\"“\—\-\(\)]+", text) if len(w)>1 and w in Emb], axis=0)

 
eo_y_preds = {}

X_train = [word_vector_featurize(p["bio"], E) for p in bios_data_train]
X_test = [word_vector_featurize(p["bio"], E) for p in bios_data_test]

print("Done featurizing")

G_train = [p["gender"] for p in bios_data_train]
G_test = [p["gender"] for p in bios_data_test]


np.random.seed(0)  
#     occs = ["painter", "journalist", "poet", "personal_trainer", "teacher", "psychologist",  "model", "interior_designer", "yoga_teacher", "nurse", "dietitian"]
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
    pkl.dump(eo_y_preds,open("results/we_tpr_sgd_preds.pkl","wb"))
