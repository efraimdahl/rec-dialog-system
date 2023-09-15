#Seperate Test and Training Data, Vectorize and save to disk to be used by seperate models.

import sklearn
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl

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
    

    df= pd.DataFrame(list(zip(labels,messages)),columns =['Label',"Text"])
    if mode=="dedupl":
        df=df.drop_duplicates(subset=['Label','Text'])

    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
    
    file = open(f"data/{mode}/X_test_raw.pkl", 'wb')
    pkl.dump(test["Text"], file)
    
    print(mode,"Splitting into training set of size ", len(train), "and test set of size ", len(test))
    vectorizer = CountVectorizer(
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(train["Text"])
    # Save vectorizer for later use
    outfile = open("models/"+mode+"/vectorizer.pkl",'wb')
    pkl.dump(vectorizer, outfile)

    # Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(test["Text"])

    y_train = train["Label"]
    y_test = test["Label"]
    feature_names = vectorizer.get_feature_names_out()
    target_names = df['Label'].unique()
    return(X_train, X_test,y_train, y_test,feature_names, target_names)

# A manual implemetation of CountVectorizer
def cntvectorizer(train,test):
    from nltk.corpus import stopwords
    stp_words = stopwords.words('english')

    words_dict = {}  # A dict contains all the words in train data
    i = 0

    train_text = train.values.tolist()
    # Build up a words dict from train set, without the stopwords
    for line in train_text:
        for word in line.split():
            if word not in words_dict.keys():
                if word not in stp_words:
                    words_dict[word] = i
                    i += 1
    vec_train = []
    for line in train_text:
        vec = np.zeros(len(words_dict.keys()))
        for word in line.split():
            if word not in stp_words:
                vec[words_dict[word]] += 1
        vec_train.append(vec)

    # Using the dict from train set for test set
    test_text = test.values.tolist()
    vec_test = []
    for line in test_text:
        vec = np.zeros(len(words_dict.keys()))
        for word in line.split():
            if word in words_dict.keys():
                if word not in stp_words:
                    vec[words_dict[word]] += 1
        vec_test.append(vec)

    return  vec_train,vec_test


filename = "dialog_acts.dat"


for mode in ["complete", "dedupl"]:
    X_train, X_test, y_train, y_test, feature_names, target_names = load_data(filename, mode)
    comps = {"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"feature_names":feature_names,"target_names":target_names}
    for name in comps.keys():
        data = comps.get(name)
        file = open("data/"+mode+"/"+name+".pkl", 'wb')
        pkl.dump(data, file)
        
