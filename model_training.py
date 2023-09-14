import pickle
from sklearn.linear_model import RidgeClassifier, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from baseline_model import KeywordClassifier

for mode in ["complete","dedupl"]:
    file1 = open("data/"+mode+"/X_train.pickle", 'rb')
    file2 = open("data/"+mode+"/y_train.pickle", 'rb')
    X_train=pickle.load(file1)
    y_train=pickle.load(file2)
    for model in ["Ridge","Linear","KNN","DecisionTree","MLP"]:
        if(model == "Ridge"):
            clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
            clf.fit(X_train, y_train)

        if (model == "Linear"):
            pass
            # clf = LinearRegression()
            # clf.fit(X_train, y_train)

        if(model == "KNN"):
            clf = KNeighborsClassifier(n_neighbors=3)
            clf.fit(X_train, y_train)

        if(model == "DecisionTree"):
            clf = DecisionTreeClassifier(random_state=0)
            clf.fit(X_train, y_train)

        if(model == "MLP"):
            pass
            # dim = len(X_train[0])
            # clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (dim, 17), random_state = 1)
        
        if model == "most_frequent":
            clf = DummyClassifier(strategy="most_frequent")
            clf.fit(X_train, y_train)

        if model == "keyword":
            clf = KeywordClassifier()
            

        outfile = open("models/"+mode+"/"+model+".pkl",'wb')
        pickle.dump(clf,outfile)


