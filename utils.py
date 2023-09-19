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

def get_distinct_values(df: pd.DataFrame, column: str) -> list:
    """
    Gets the distinct values from the specified column in the dataframe.

    Args:
        df (pd.DataFrame): Dataframe to get the distinct values from
        column (str): Column to get the distinct values from

    Returns:
        list: List of distinct values
    """
    return df[column].unique().tolist()

def main():
    data = load_data('restaurant_info.csv')
    print("Price range options: ", get_distinct_values(data, 'pricerange'))
    print("Area options: ", get_distinct_values(data, 'area'))
    print("Food options: ", get_distinct_values(data, 'food'))
    
    
if __name__ == '__main__':
    main()