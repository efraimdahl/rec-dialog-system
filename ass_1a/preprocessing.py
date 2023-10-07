import sklearn
import numpy as np
import pandas as pd
import pickle
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

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
    
    result = {}
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
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    with open('label_encoder_model.pkl', 'wb') as file:
        pickle.dump(label_encoder, file)
    df = pd.DataFrame(list(zip(labels,messages)), columns = ['Label',"Text"])
    train, test = sklearn.model_selection.train_test_split(df, test_size=0.2, stratify=df["Label"])
    
    # Remove duplicates for deduplicated mode
    if mode == "deduplicated":
        train=train.drop_duplicates(subset=['Label','Text'])
        test=test.drop_duplicates(subset=['Label','Text'])
    
    # Save raw train/test data for keyword classifier
    result["X_train_raw"] = train["Text"]
    result["X_test_raw"] = test["Text"]
    
    print(mode,"Splitting into training set of size ", len(train), "and test set of size ", len(test))
    
    vectorizer = CountVectorizer(
        stop_words="english"
    )
    result["X_train"] = vectorizer.fit_transform(train["Text"])
    
    # Save vectorizer for later use
    result["vectorizer"] = vectorizer

    # Extracting features from the test data using the same vectorizer
    result["X_test"] = vectorizer.transform(test["Text"])

    result["y_train"] = train["Label"]
    result["y_test"] = test["Label"]
    result["feature_names"] = vectorizer.get_feature_names_out()
    result["target_names"] = df['Label'].unique()
    return result

def prepare_data(filename):
    """
    Loads and prepares all the data for training, split by complete and deduplicated modes.

    Parameters
    ----------
    filename : file to be loaded

    Returns
    -------
    data : dictionary containing all the data

    """
    print("Preparing Data")
    data = {}
    for mode in ["complete", "deduplicated"]:
        res = load_data(filename, mode)
        data[mode] = res
    return data

if __name__ == "__main__":
    filename = "data/dialog_acts.dat"
    data = prepare_data(filename)
    print(data["complete"]["X_train"].shape)
    print(data["complete"]["X_test"].shape)
    print(data["complete"]["y_train"].shape)
    print(data["complete"]["y_test"].shape)
