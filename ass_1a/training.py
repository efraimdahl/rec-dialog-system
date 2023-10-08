from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from ass_1a.keyword_model import KeywordClassifier
from sklearn.neural_network import MLPClassifier

def train_model(data, model):
    """
    Trains a single model on the training data 

    Parameters
    ----------
    model : model to be trained
    
    Returns
    -------
    model : a trained model object.

    """
    X_train=data["X_train"]
    y_train=data["y_train"]
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

    if model == "MLP":
        clf = MLPClassifier(hidden_layer_sizes=(571, 15), max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
    return clf

def train_models(data):
    """Trains all models using the train_model function"""
    modes = ["complete","deduplicated"]
    models = {modes[0] : {},modes[1] : {}}
    for mode in modes:
        for model in ["Ridge","KNN","DecisionTree", "most_frequent", "keyword", "MLP"]:
            models[mode][model] = train_model(data[mode], model)
    print("Completed training models")
    return models