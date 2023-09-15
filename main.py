from data_prep import prepare_data
from model_training import train_models
from model_eval import evaluate_models


def main():
    prepare_data()
    train_models()
    evaluate_models()
            
if __name__ == "__main__":
    main()