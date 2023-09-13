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
    limit = 0
    for line in data:
        line = line.rstrip("\n")  
        label = line.split(" ")[0]
        message = line.lstrip(label)
        messages.append(message)
        labels.append(label)
        limit+=1
    vectorizer = TfidfVectorizer(
        sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english"
    )
    df= pd.DataFrame(list(zip(labels,messages)),columns =['Label',"Text"])
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
    X_train = vectorizer.fit_transform(train["Text"])

    # Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(test["Text"])
    y_train = train["Label"]
    y_test = test["Label"]
    feature_names = vectorizer.get_feature_names_out()
    target_names = df['Label'].unique()
    return(X_train, X_test,y_train, y_test,feature_names, target_names)

filename = "dialog_acts.dat"
X_train, X_test, y_train, y_test, feature_names, target_names = load_data(filename)

clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(target_names)

fig, ax = plt.subplots(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax,labels=target_names)
_ = ax.set_title(
    f"Confusion Matrix for Restaurant Dialog Classifier"
)
fig.savefig("Base Confusion Matrix.png")
