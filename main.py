import sklearn
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.linear_model import RidgeClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
import baseline_model
import sklearn.dummy

def load_data(filename):
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
    return df


filename = "dialog_acts.dat"
df = load_data(filename)

train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
vectorizer = TfidfVectorizer(
    sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
)

X_train = vectorizer.fit_transform(train["Text"])
# Extracting features from the test data using the same vectorizer
X_test = vectorizer.transform(test["Text"])
y_train = train["Label"]
y_test = test["Label"]
feature_names = vectorizer.get_feature_names_out()
target_names = df['Label'].unique()

# Most frequent baseline

freqmodel = sklearn.dummy.DummyClassifier(strategy="most_frequent")
freqmodel.fit(X_train,y_train)
pred_freq = freqmodel.predict(X_test)
print(freqmodel.score(X_test, y_test))

# Keyword matching baseline

keywordmodel = baseline_model.KeywordClassifier()
pred_keyword = keywordmodel.predict(test["Text"])
print(keywordmodel.score(test["Text"],y_test))

# Ridge classifier
clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)


fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y_test, pred_freq, ax=ax,labels=target_names)
_ = ax.set_title(
    f"Confusion Matrix using Most Frequent model"
)
fig.savefig("Most frequent model Confusion Matrix.png")

fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y_test, pred_keyword, ax=ax,labels=target_names)
_ = ax.set_title(
    f"Confusion Matrix using Keyword Matching model"
)
fig.savefig("Keyword model Confusion Matrix.png")

fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax,labels=target_names)
_ = ax.set_title(
    f"Confusion Matrix using Ridge classifier"
)
fig.savefig("Ridge model Confusion Matrix.png")
