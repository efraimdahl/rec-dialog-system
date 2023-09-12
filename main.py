import sklearn
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

stp_words = stopwords.words('english')

def load_data(filename):
    with open(filename) as f:
        data = [line.rstrip("\n") for line in f.readlines()]
    labels, messages = [], []
    for line in data:
        line = line.rstrip("\n")  
        label = line.split(" ")[0]
        message = line.lstrip(label)
        messages.append(message)
        labels.append(label)
    #vectorizer = TfidfVectorizer(
    #    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    #)
    df= pd.DataFrame(list(zip(labels,messages)),columns =['Label',"Text"])
    #Act_Dict : {"ack" : 0, "affirm" : 1, "bye" : 2, "confirm" : 3 , "deny" : 4, "hello" : 5, "inform" : 6, "negate" : 7, "null" : 8, "repeat" : 9, "null" : 10, "repeat": 11, "reqalts" : 12, "reqmore" : 13, "request" : 14, "restart" : 15, "thankyou" : 16}

    #print(df)
    #print(vectorList)
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
    words_dict = {} #A dict contains all the words in train data
    i = 0

    train_text = train['Text'].values.tolist()

    for line in train_text:
        for word in line.split():
            if word not in words_dict.keys():
                if word not in stp_words:
                    words_dict[word]=i
                    i += 1
    vec_train = []
    for line in train_text:
        vec = np.zeros(len(words_dict.keys()))
        for word in line.split():
            if word not in stp_words:
                vec[words_dict[word]] += 1
        vec_train.append(vec)
    train['Vector'] = vec_train

    test_text = test['Text'].values.tolist()
    vec_test = []
    for line in test_text:
        vec = np.zeros(len(words_dict.keys()))
        for word in line.split():
            if word in words_dict.keys():
                if word not in stp_words:
                    vec[words_dict[word]] += 1
        vec_test.append(vec)
    test['Vector'] = vec_test

    return(train,test,df["Label"].unique())




filename = "dialog_acts.dat"
train,test,targets = load_data(filename)
X_train,y_train,X_test,y_test = train["Vector"],train["Label"],test["Vector"],test["Label"]

'''

df=pd.read_csv('dialog_acts.dat',sep=' ')

print(df)

temp2=df.X0.str.split(' ',expand=True)
del df['X0']
print(len(df),df.columns)
'''

