import pickle
from sklearn.linear_model import RidgeClassifier

for mode in ["complete","dedupl"]:
    file1 = open("data/"+mode+"/X_train.pickle", 'rb')
    file2 = open("data/"+mode+"/y_train.pickle", 'rb')
    X_train=pickle.load(file1)
    y_train=pickle.load(file2)
    for model in ["Ridge"]:
        if(model=="Ridge"):
            clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
            clf.fit(X_train, y_train)
        outfile = open("models/"+mode+"/"+model+".pkl",'wb')
        pickle.dump(model,outfile)

