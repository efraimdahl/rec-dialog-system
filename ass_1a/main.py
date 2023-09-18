from preprocessing import prepare_data
from training import train_models
from evaluation import evaluate_models


def main():
    prepare_data()
    train_models()
    evaluate_models()

if __name__ == "__main__":
    main()