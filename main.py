class State:
    def __init__(self, name="hello"):
        self.name = name
        
        self.knowledge_base = dict(
        price = None,
        area = None,
        food = None
        )
        
        self.current_response = "Hi, welcome to the restaurant system! You can ask for restaurants by area , price range or food type . How may I help you?"
        
    def __str__(self):
        return self.name
    
    def update_knowledge(knowledge_base: dict, new_data: dict) -> None:
        """Updates the knowledge base with new information.

        Args:
            knowledge_base (dict): A dict containing price, area, and food information as keys.
            new_data (dict): A dict containing price, area, and food information as keys.
        """
        
        pass
    
    
    def generate_response(self, dialog_act: str, knowledge_base: dict) -> str:
        """Generates the appropriate response based on the speech act and .

        Args:
            speech_act (str): _description_
            knowledge_base (dict): _description_

        Returns:
            str: _description_
        """
        
        pass
    
    def update_state(self) -> None:
        """Updates the state of the dialog based on the current state and the user's input.
        """
        pass

        
def determine_speech_act(user_input: str) -> str:
    """Uses KNN to classify the user's input to determine the corresponding speech act.

    Args:
        user_input (str): A user response.

    Returns:
        str: The speech act corresponding to the user_input.
    """
    # TODO: 
    # - Import trained KNN model (more efficient to to this 
    #   just once outside of this function, idk where though)
    # - Classify and return user_input using KNN model
    
    pass

def keyword_match(user_input: str) -> dict:
    """Uses keyword matching to determine what information is passed along.

    Args:
        user_input (str): A user response.

    Returns:
        dict: A dict containing price, area, and food information as keys.
    """
    # TODO:
    # - 
    
    pass
    


def generate_response(current_state: State, speech_act: str, knowledge_base: dict) -> str:
    """Generates a response based on the current state, the speech act, and the knowledge base.

    Args:
        current_state (State): The current state of the dialog
        speech_act (str): The speech act corresponding to the user_input.
        knowledge_base (dict): A dict containing price, area, and food information as keys.

    Returns:
        str: The response to be returned.
    """
    
    pass

def state_transition(current_state: State, user_input: str) -> tuple(State, str):
    """
    Using an input state and an action, determines the 
    next state and chatbot response.
    
    Args:
        current_state (State): The current state of the dialog
        user_input (str): The most recent user response.

    Returns:
        str: String to be returned.
    """
    #TODO:
    #- Classify user_input's speech act and keyword_match any information passed along.
    #- Based on the speech act and the (new) information passed along, determine bot response.
    
    speech_act = determine_speech_act(user_input)
    new_data = keyword_match(user_input)
    knowledge_base = current_state.update_knowledge(knowledge_base, new_data)
    
    response = generate_response(current_state, speech_act, knowledge_base)
    
    
    new_state = current_state.update_state
    
    return tuple(new_state, response)

def main():
    state = State()
    
    pass

if __name__ == '__main__':
    main()