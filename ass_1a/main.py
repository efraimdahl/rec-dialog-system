from preprocessing import prepare_data
from training import train_models
from evaluation import evaluate_models


def main():
    filename = "data/dialog_acts.dat"
    data = prepare_data(filename)
    models = train_models(data)
    print(models.keys())
    evaluate_models(models,data)
    # Evaluation of models is saved in the models map.

if __name__ == "__main__":
    main()