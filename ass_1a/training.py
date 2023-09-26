from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from keyword_model import KeywordClassifier

def train_models(data):
    """
    A list of different models is trained on the training data 

    Parameters
    ----------
    data : dictionary which contains the training data for each mode (complete,deduplicated).
    
    Returns
    -------
    models : a dictionary containing the trained models (key: name, val: trained model object).

    """
    modes = ["complete","deduplicated"]
    models = {modes[0] : {},modes[1] : {}}
    for mode in modes:
        X_train=data[mode]["X_train"]
        y_train=data[mode]["y_train"]
        for model in ["Ridge","KNN","DecisionTree", "most_frequent", "keyword"]:
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
                # No fitting
                
            models[mode][model] = clf
    print("Completed training models")
    return models