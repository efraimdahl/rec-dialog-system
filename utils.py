import pandas as pd

def load_data(filename: str) -> pd.DataFrame:
    """
    Loads the data from the csv file and returns a list of lists.

    Args:
        filename (str): path to the csv file

    Returns:
        pd.DataFrame: Dataframe corresponding to the csv file
    """
    df = pd.read_csv(filename, header=0)
    return df


def main():
    data = load_data('restaurant_info.csv')
    print(data)
    
    
if __name__ == '__main__':
    main()