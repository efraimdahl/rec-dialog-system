import sklearn
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer

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
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    df= pd.DataFrame(list(zip(labels,messages)),columns =['Label',"Text"])
    #Act_Dict : {"ack" : 0, "affirm" : 1, "bye" : 2, "confirm" : 3 , "deny" : 4, "hello" : 5, "inform" : 6, "negate" : 7, "null" : 8, "repeat" : 9, "null" : 10, "repeat": 11, "reqalts" : 12, "reqmore" : 13, "request" : 14, "restart" : 15, "thankyou" : 16}
    vectorList = vectorizer.fit_transform(messages)
    df["Vector"]=vectorList
    print(vectorList)
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
    print(test)
    return(train,test,df["Label"].unique())


filename = "dialog_acts.dat"
train,test,targets = load_data(filename)
X_train,y_train,X_test,y_test = train["Vector"],train["Label"],test["Vector"],test["Label"]

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


fig, ax = plt.subplots(figsize=(10, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax)
ax.xaxis.set_ticklabels(targets)
ax.yaxis.set_ticklabels(targets)
_ = ax.set_title(
    f"Confusion Matrix for Restaurant Dialog Classifier"
)
'''

df=pd.read_csv('dialog_acts.dat',sep=' ')

print(df)

temp2=df.X0.str.split(' ',expand=True)
del df['X0']
print(len(df),df.columns)
'''

