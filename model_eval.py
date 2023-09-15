import pickle as pkl
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator

def predict_difficult_instance(model, vectorizer):
    difficult_instances = [
        "Not really what im looking for, what about korean food", # Might be classified as negate, should be reqalt
        "no id rather find a moderately priced restaurant",  # Might be classified negate, should be reqalt
        "amazing thank you very much goodbye",      # can represent two labels
        "goodbye thank you for your help",          # can represent two labels
        "I am looking for an Italian restoration",  # restoration instead of restaurant
    ]

    for inst in difficult_instances:
        inst_tf = vectorizer.transform([inst]) if repr(model) != "KeywordClassifier()" else [inst]
        print(f"Sentence: {inst}\tPredicted label: {model.predict(inst_tf)}")
        
    return


for mode in ["complete","dedupl"]:
    file1 = open("data/"+mode+"/X_test.pkl", 'rb')
    file2 = open("data/"+mode+"/y_test.pkl", 'rb')
    file3 = open("data/"+"complete"+"/target_names.pkl", 'rb')
    file4 = open("models/"+mode+"/vectorizer.pkl",'rb')
    file5 = open("data/"+mode+"/X_test_raw.pkl", 'rb')
    
    X_test=pkl.load(file1)
    y_test=pkl.load(file2)
    target_names=pkl.load(file3)
    vectorizer = pkl.load(file4)
    X_test_raw=pkl.load(file5)
    
    
    for model in ["Ridge", "KNN", "DecisionTree", "most_frequent", "keyword"]:
        modelfile = open("models/"+mode+"/"+model+".pkl",'rb')
        pickled_model = pkl.load(modelfile)
        if model == "keyword":
            pred = pickled_model.predict(X_test_raw)
        else:
            pred = pickled_model.predict(X_test)
        fig, ax = plt.subplots(figsize=(10,10))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax,labels=target_names)
        _ = ax.set_title(
            f"Confusion Matrix for Restaurant Dialog Classifier model=${model}/data=${mode}"
        )
        fig.savefig("results/"+model+"_"+mode+".png")
        
        
        # Print classification report
        print(f"\n\nClassification report for model {model} trained on {mode}")
        print(y_test, target_names)
        print(classification_report(y_test, pred, target_names=target_names, zero_division=0))
        
        
        
        print(f"\nPredicting difficult instances for model {model} trained on {mode}")
        predict_difficult_instance(pickled_model, vectorizer)
        
        


