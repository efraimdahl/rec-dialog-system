from preprocessing import prepare_data
from training import train_models
from evaluation import evaluate_models


def main():
    """
    Does everything for assignment 1a: loading the data, training the models and evaluating the models
    """
    filename = "data/dialog_acts.dat"
    data = prepare_data(filename)
    models = train_models(data)
    evaluate_models(models,data)
    # Evaluation of models is saved in the results map.

if __name__ == "__main__":
    main()