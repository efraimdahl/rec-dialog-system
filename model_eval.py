import pickle
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

for mode in ["complete","dedupl"]:
    file1 = open("data/"+mode+"/X_test.pickle", 'rb')
    file2 = open("data/"+mode+"/y_test.pickle", 'rb')
    file3 = open("data/"+"complete"+"/target_names.pickle", 'rb')
    X_test=pickle.load(file1)
    y_test=pickle.load(file2)
    target_names=pickle.load(file3)
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

