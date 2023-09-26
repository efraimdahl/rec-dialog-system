import sklearn
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer
import pickle as pkl

def load_data(filename:str, mode:str) -> tuple:
    """Load data from file and split into training and test set.

    Args:
        filename (str): Filename as a string.
        mode (str): Mode of data, either "complete" or "deduplicated".

    Returns:
        tuple: Tuple containing the training and test data.
    """
    with open(filename) as f:
        data = [line.rstrip("\n") for line in f.readlines()]
    
    # Process the dataset
    labels, messages = [], []
    limit = 0
    for line in data:
        line = line.rstrip("\n")  
        label = line.split(" ")[0]
        message = line.lstrip(label)
        messages.append(message)
        labels.append(label)
        limit += 1
    
    df = pd.DataFrame(list(zip(labels,messages)), columns = ['Label',"Text"])
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, stratify=df["Label"])
    
    # Remove duplicates for deduplicated mode
    if mode == "deduplicated":
        train=train.drop_duplicates(subset=['Label','Text'])
        test=test.drop_duplicates(subset=['Label','Text'])
    
    # Save raw test data for keyword classifier
    file = open(f"ass_1a/data/{mode}/X_test_raw.pkl", 'wb')
    pkl.dump(test["Text"], file)
    
    print(mode,"Splitting into training set of size ", len(train), "and test set of size ", len(test))
    
    vectorizer = CountVectorizer(
        stop_words="english"
    )
    X_train = vectorizer.fit_transform(train["Text"])
    
    # Save vectorizer for later use
    outfile = open("ass_1a/models/"+mode+"/vectorizer.pkl",'wb')
    pkl.dump(vectorizer, outfile)

    # Extracting features from the test data using the same vectorizer
    X_test = vectorizer.transform(test["Text"])

    y_train = train["Label"]
    y_test = test["Label"]
    feature_names = vectorizer.get_feature_names_out()
    target_names = df['Label'].unique()
    return (X_train, X_test, y_train, y_test, feature_names, target_names)

def prepare_data():
    print("Preparing Data ")
    filename = "data/dialog_acts.dat"
    for mode in ["complete", "deduplicated"]:
        X_train, X_test, y_train, y_test, feature_names, target_names = load_data(filename, mode)
        comps = {"X_train":X_train,"X_test":X_test,"y_train":y_train,"y_test":y_test,"feature_names":feature_names,"target_names":target_names}
        for name in comps.keys():
            data = comps.get(name)
            file = open("ass_1a/data/"+mode+"/"+name+".pkl", 'wb')
            pkl.dump(data, file)

if __name__ == "__main__":
    prepare_data()
