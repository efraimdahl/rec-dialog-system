from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from keyword_model import KeywordClassifier

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
        inst_tf = vectorizer.transform([inst]) if not isinstance(model,KeywordClassifier) else [inst]
        results += f"Sentence: {inst}\tPredicted label: {model.predict(inst_tf)}\n"
    print(results if print_string else "")
    return results

def evaluate_models(models,data):
    for mode in ["complete","deduplicated"]:    
        for model_name,model in models[mode].items():
           
            if model_name == "keyword":
                pred = model.predict(data[mode]["X_test_raw"])
            else:
                pred = model.predict(data[mode]["X_test"])
                
            # Plot and save confusion matrix
            fig, ax = plt.subplots(figsize=(10,10))
            ConfusionMatrixDisplay.from_predictions(data[mode]["y_test"], pred, ax=ax,labels=data[mode]["target_names"])
            _ = ax.set_title(
                f"Confusion Matrix for Restaurant Dialog Classifier model={model_name}/data={mode}"
            )
<<<<<<< Updated upstream
            fig.savefig(f"results/{model}_{mode}.png")
            
            # Print classification report
            ordered_labels = ["ack", "affirm", "bye", "confirm", "deny", "hello", "inform", "negate", "null", "repeat", "reqalts", "require", "request", "restart", "thankyou"]
            print(f"\n\nClassification report for model {model} trained on {mode}")
            print(classification_report(y_test, pred, target_names=ordered_labels, zero_division=0))
            
            # Save classification report to log file
            with open(f"results/{model}_{mode}.txt", "w") as f:
                f.write(f"Classification report for model {model} trained on {mode}\n")
                f.write(classification_report(y_test, pred, target_names=ordered_labels, zero_division=0))
=======
            fig.savefig(f"ass_1a/results/{model_name}_{mode}.png")
            
            # Save classification report to log file
            with open(f"ass_1a/results/{model_name}_{mode}.txt", "w") as f:
                f.write(f"\n\nClassification report for model {model_name} trained on {mode}\n")
                f.write(classification_report(data[mode]["y_test"], pred, target_names=data[mode]["target_names"], zero_division=0))
>>>>>>> Stashed changes
            
            # Print difficult instances
            print(f"\nPredicting difficult instances for model {model_name} trained on {mode}")
            difficult_instances = predict_difficult_instance(model, data[mode]["vectorizer"])
            
            # Save difficult instances to log file
            with open(f"ass_1a/results/{model_name}_{mode}.txt", "a") as f:
                f.write(f"Predicting difficult instances for model {model_name} trained on {mode}\n")
                f.write(difficult_instances)


