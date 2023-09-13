#Seperate Test and Training Data, Vectorize and save to disk to be used by seperate models.

import sklearn
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
import pickle

def load_data(filename,mode):
    with open(filename) as f:
        data = [line.rstrip("\n") for line in f.readlines()]
    labels, messages = [], []
    limit = 0
    for line in data:
        line = line.rstrip("\n")  
        label = line.split(" ")[0]
        message = line.lstrip(label)
        messages.append(message)
        labels.append(label)
        limit+=1
    
    vectorizer = CountVectorizer(
        stop_words="english"
    )
    df= pd.DataFrame(list(zip(labels,messages)),columns =['Label',"Text"])
    if mode=="dedupl":
        df=df.drop_duplicates(subset=['Label','Text'])

    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
    print(mode,"Splitting into training set of size ", len(train), "and test set of size ", len(test))
    X_train = vectorizer.fit_transform(train["Text"])

    # Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(test["Text"])
    y_train = train["Label"]
    y_test = test["Label"]
    feature_names = vectorizer.get_feature_names_out()
    target_names = df['Label'].unique()
    return(X_train, X_test,y_train, y_test,feature_names, target_names)

filename = "dialog_acts.dat"


for mode in ["complete","dedupl"]:
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data(filename, mode)
    comps = {"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"feature_names":feature_names,"target_names":target_names}
    for name in comps.keys():
        data = comps.get(name)
        file = open("data/"+mode+"/"+name+".pickle", 'wb')
        pickle.dump(data, file)