import pandas as pd
import random
from config import *

import os
import sys 
import time
from gtts import gTTS
import pygame


def clear_line() -> None:
    """Clears the current line in the console.
    """
    sys.stdout.write("\033[F")
    sys.stdout.write("\033[K")
    
def take_user_input() -> str:
    """Helper function to get user input and print it to the console.
    Also integrates configurability relating to response style.
    """
    user_input = input("Type your response: ")
    clear_line()
    if COLORED_OUTPUT:
        sentence = "\033[1;33;40mUser: \033[0m" + "\033[1;36;40m" + user_input + "\033[0m"
    else:
        sentence = "User: " + user_input
    print(sentence)
    return user_input
    

def add_database_column(file_name: str, col_name: str, options: list) -> None:
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
    # Message preprocessing
    tts = gTTS(message, lang='en')
    tts.save("output.mp3")
    pygame.init()
    message = message.upper() if ALL_CAPS_RESPONSE else message
    
    if COLORED_OUTPUT:
        message = "\033[1;32;40mChatbot: \033[0m" + "\033[1;36;40m" + message + "\033[0m"
    else:
        message = "Chatbot: " + message
        
    
    if RESPONSE_DELAY > 0:
        print("Thinking...")
        time.sleep(RESPONSE_DELAY)
        clear_line()
        
    if TYPING_SPEED_DELAY > 0:
        for word in (message).split():
            sys.stdout.write(word + " ")
            sys.stdout.flush()
            time.sleep(TYPING_SPEED_DELAY)
        sys.stdout.write("\n")
    else:
        print(message)
    pygame.mixer.music.load("output.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(1)
    pygame.mixer.quit()
    os.remove("output.mp3")
    
def main():
    filename = "data/restaurant_info.csv"
    add_database_column(filename, "food_quality", ["good", "bad", "average"])
    add_database_column(filename, "crowdedness" , ["crowded", "not crowded", "average"])
    add_database_column(filename, "length_of_stay", ["long", "short", "average"])
    
if __name__ == "__main__":
    main()