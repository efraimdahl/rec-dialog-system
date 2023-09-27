import pandas as pd
import random
from config import *
from time import sleep

def add_database_column(file_name: str, col_name: str, options: list[str]) -> None:
    """Adds a column to a database file if it does not exist yet, 
    randomly fills it using the options provided and saves the file.

    Args:
        file_name (str): The file name of the database file.
        col_name (str): The name of the column to be added.
        options (list[str]): The options to be added to the column.
    """
    df = pd.read_csv(file_name)
    if col_name not in df.columns:
        df[col_name] = [random.choice(options) for _ in range(len(df))]
        df.to_csv(file_name, index=False)
        
def chatbot_print(message: str) -> None:
    """Helper function to make chatbot messages stand out from debugging messages.
    Also integrates configurability relating to response style. 

    Args:
        message (str): The message to be printed.
    """
    message = message.upper() if ALL_CAPS_RESPONSE else message
    
    if RESPONSE_DELAY > 0:
        print("Thinking...")
        sleep(RESPONSE_DELAY)
        print("\033[A                             \033[A")
    print("Chatbot: " + message)
    
def main():
    filename = "restaurant_info.csv"
    add_database_column(filename, "food_quality", ["good", "bad", "average"])
    add_database_column(filename, "crowdedness" , ["crowded", "not crowded", "average"])
    add_database_column(filename, "length_of_stay", ["long", "short", "average"])
    
if __name__ == "__main__":
    main()