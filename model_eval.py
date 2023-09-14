import pickle
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
        inst_tf = vectorizer.transform([inst])
        print(f"Sentence: {inst}\tPredicted label: {model.predict(inst_tf)}")
        
    return


for mode in ["complete","dedupl"]:
    file1 = open("data/"+mode+"/X_test.pickle", 'rb')
    file2 = open("data/"+mode+"/y_test.pickle", 'rb')
    file3 = open("data/"+"complete"+"/target_names.pickle", 'rb')
    file4 = open("models/"+mode+"/vectorizer.pkl",'rb')
    X_test=pickle.load(file1)
    y_test=pickle.load(file2)
    target_names=pickle.load(file3)
    vectorizer = pickle.load(file4)
    
    
    for model in ["Ridge","KNN", "DecisionTree"]:
        modelfile = open("models/"+mode+"/"+model+".pkl",'rb')
        pickled_model = pickle.load(modelfile)
        pred = pickled_model.predict(X_test)
        fig, ax = plt.subplots(figsize=(10,10))
        ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax,labels=target_names)
        _ = ax.set_title(
            f"Confusion Matrix for Restaurant Dialog Classifier model=${model}/data=${mode}"
        )
        fig.savefig("results/"+model+"_"+mode+".png")
        
        
        # Print classification report
        print(f"\n\nClassification report for model {model} trained on {mode}")
        print(classification_report(y_test, pred, target_names=target_names, zero_division=0))
        
        
        
        print(f"\nPredicting difficult instances for model {model} trained on {mode}")
        predict_difficult_instance(pickled_model, vectorizer)
        
        


