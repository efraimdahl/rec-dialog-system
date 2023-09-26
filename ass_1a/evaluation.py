import pickle as pkl
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

def predict_difficult_instance(model, vectorizer, print_string=True) -> str:
    difficult_instances = [
        "Not really what im looking for, what about korean food", # Might be classified as negate, should be reqalt
        "no id rather find a moderately priced restaurant",  # Might be classified negate, should be reqalt
        "amazing thank you very much goodbye",      # can represent two labels
        "goodbye thank you for your help",          # can represent two labels
        "I am looking for an Italian restoration",  # restoration instead of restaurant
    ]

    results = ""
    for inst in difficult_instances:
        inst_tf = vectorizer.transform([inst]) if repr(model) != "KeywordClassifier()" else [inst]
        results += f"Sentence: {inst}\tPredicted label: {model.predict(inst_tf)}\n"
    print(results if print_string else "")
    return results

def evaluate_models():
    for mode in ["complete","deduplicated"]:
        # Load all pickle files
        X_test_file = open(f"data/{mode}/X_test.pkl", 'rb')
        y_test_file = open(f"data/{mode}/y_test.pkl", 'rb')
        labels_file = open(f"data/complete/target_names.pkl", 'rb')
        vectorizer_file = open(f"models/{mode}/vectorizer.pkl",'rb')
        X_test_raw_file = open(f"data/{mode}/X_test_raw.pkl", 'rb')
        
        X_test=pkl.load(X_test_file)
        y_test=pkl.load(y_test_file)
        target_names=pkl.load(labels_file)
        vectorizer = pkl.load(vectorizer_file)
        X_test_raw=pkl.load(X_test_raw_file)
        
        
        for model in ["Ridge", "KNN", "DecisionTree", "most_frequent", "keyword"]:
            modelfile = open(f"models/{mode}/{model}.pkl",'rb')
            clf = pkl.load(modelfile)
            
            if model == "keyword":
                pred = clf.predict(X_test_raw)
            else:
                pred = clf.predict(X_test)
                
            # Plot and save confusion matrix
            fig, ax = plt.subplots(figsize=(10,10))
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax,labels=target_names)
            _ = ax.set_title(
                f"Confusion Matrix for Restaurant Dialog Classifier model={model}/data={mode}"
            )
            fig.savefig(f"results/{model}_{mode}.png")
            
            # Print classification report
            ordered_labels = ["ack", "affirm", "bye", "confirm", "deny", "hello", "inform", "negate", "null", "repeat", "reqalts", "require", "request", "restart", "thankyou"]
            print(f"\n\nClassification report for model {model} trained on {mode}")
            print(classification_report(y_test, pred, target_names=ordered_labels, zero_division=0))
            
            # Save classification report to log file
            with open(f"results/{model}_{mode}.txt", "w") as f:
                f.write(f"Classification report for model {model} trained on {mode}\n")
                f.write(classification_report(y_test, pred, target_names=ordered_labels, zero_division=0))
            
            # Print difficult instances
            print(f"\nPredicting difficult instances for model {model} trained on {mode}")
            difficult_instances = predict_difficult_instance(clf, vectorizer)
            
            # Save difficult instances to log file
            with open(f"results/{model}_{mode}.txt", "a") as f:
                f.write(f"Predicting difficult instances for model {model} trained on {mode}\n")
                f.write(difficult_instances)


