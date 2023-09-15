import pickle as pkl
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from keyword_classifier import KeywordClassifier

def train_models():
    for mode in ["complete","dedupl"]:
        file1 = open("data/"+mode+"/X_train.pkl", 'rb')
        file2 = open("data/"+mode+"/y_train.pkl", 'rb')
        X_train=pkl.load(file1)
        y_train=pkl.load(file2)
        for model in ["Ridge","Linear","KNN","DecisionTree","MLP", "most_frequent", "keyword"]:
            if(model == "Ridge"):
                clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
                clf.fit(X_train, y_train)

            if(model == "KNN"):
                clf = KNeighborsClassifier(n_neighbors=3)
                clf.fit(X_train, y_train)

            if(model == "DecisionTree"):
                clf = DecisionTreeClassifier(random_state=0)
                clf.fit(X_train, y_train)
            
            if model == "most_frequent":
                clf = DummyClassifier(strategy="most_frequent")
                clf.fit(X_train, y_train)

            if model == "keyword":
                clf = KeywordClassifier()
                

            outfile = open("models/"+mode+"/"+model+".pkl",'wb')
            pkl.dump(clf,outfile)

if __name__ == "__main__":
    train_models()