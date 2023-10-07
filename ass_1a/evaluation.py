from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import pickle
import numpy as np
from keyword_model import KeywordClassifier
from sklearn.preprocessing import LabelEncoder

with open('label_encoder_model.pkl', 'rb') as file:
    loaded_label_encoder = pickle.load(file)
def predict_difficult_instance(model, vectorizer, print_string=True) -> str:
    """
    A couple difficult sentences were selected to test each model's performance on, see report.

    Parameters
    ----------
    model : the model used to predict labels
    vectorizer : vectorizer used to vectorize inputs
    print_string : bool, whether to print the ouptut on top of returning it.

    Returns
    -------
    str
        A string describing the labels assigned to each sentence by the model

    """
    
    difficult_instances = [
        "Not really what im looking for, what about korean food", # Might be classified as negate, should be reqalt
        "no id rather find a moderately priced restaurant",  # Might be classified negate, should be reqalt
        "amazing thank you very much goodbye",      # can represent two labels
        "goodbye thank you for your help",          # can represent two labels
        "I am looking for an Italian restoration",  # restoration instead of restaurant
    ]

    results = ""
    for inst in difficult_instances:
        inst_tf = vectorizer.transform([inst]) if not isinstance(model,KeywordClassifier) else [inst]
        results += f"Sentence: {inst}\tPredicted label: {model.predict(inst_tf)}\n"
    print(results if print_string else "")
    return results

def evaluate_models(models,data):
    """
    Evaluates model performance and generates confusion matrices, classification reports and logs difficult isntances.

    Parameters
    ----------
    models : a dictionary containing the models to be evaluated (key: name, val: trained model).
    data : a dictionary containing the data (key: mode, val: data for that mode)

    Returns
    -------
    None.

    """
    for mode in ["complete","deduplicated"]:    
        for model_name,model in models[mode].items():
           
            if model_name == "keyword":
                pred = model.predict(data[mode]["X_test_raw"])
            else:
                pred = model.predict(data[mode]["X_test"])
                
            # Plot and save confusion matrix
            fig, ax = plt.subplots(figsize=(10,10))
            ordered_labels = ["ack", "affirm", "bye", "confirm", "deny", "hello", "inform", "negate", "null", "repeat", "reqalts", "require", "request", "restart", "thankyou"]
            pred = list(pred)
            if isinstance(pred[0], np.int64):
                pred = loaded_label_encoder.inverse_transform(pred)
            y_test = list(data[mode]["y_test"])
            y_test= loaded_label_encoder.inverse_transform(y_test)
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=ax, labels=ordered_labels)
            _ = ax.set_title(
                f"Confusion Matrix for Restaurant Dialog Classifier model={model_name}/data={mode}"
            )
            fig.savefig(f"ass_1a/results/{model_name}_{mode}.png")

            # Save classification report to log file

            with open(f"ass_1a/results/{model_name}_{mode}.txt", "w") as f:
                f.write(f"\n\nClassification report for model {model_name} trained on {mode}\n")
                f.write(classification_report(y_test, pred, target_names=ordered_labels, zero_division=0))
            
            # Print difficult instances
            print(f"\nPredicting difficult instances for model {model_name} trained on {mode}")
            difficult_instances = predict_difficult_instance(model, data[mode]["vectorizer"])
            
            # Save difficult instances to log file
            with open(f"ass_1a/results/{model_name}_{mode}.txt", "a") as f:
                f.write(f"Predicting difficult instances for model {model_name} trained on {mode}\n")
                f.write(difficult_instances)


