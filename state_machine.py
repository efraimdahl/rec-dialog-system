from statemachine import StateMachine, State
import pandas as pd
from text_parser import TextParser
from statemachine.contrib.diagram import DotGraphMachine
import warnings
warnings.filterwarnings("ignore")

from typing import Union, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import ClassifierMixin

from ass_1a.keyword_model import KeywordClassifier
from ass_1a.training import train_model
from ass_1a.preprocessing import prepare_data
from utils import chatbot_print, take_user_input
from config import *
import json
import time
import random


class RestaurantAgent(StateMachine):
    """State machine that runs a restaurant recommondation dialog based on desired area, priceRange and cuisine.
    """
    # STATES
    hello = State(initial=True)
    state_preferences = State()
    process_preferences = State()
    ask_levenshtein = State()
    ask_area = State()
    ask_priceRange = State()
    ask_foodType = State()
    ask_qualifier = State()
    return_restaurant = State()
    no_restaurant_found = State()
    preference_reasoning = State()
    process_alternative = State()
    give_information = State()
    cant_give_information = State()
    completed = State(final=True)
    
    # TRANSITIONS
    start_processing = hello.to(state_preferences)
    
    receive_input = (
        state_preferences.to(process_preferences, cond="input_received")
        | ask_levenshtein.to(process_preferences, cond="input_received")
        | ask_area.to(process_preferences, cond= "input_received")
        | ask_priceRange.to(process_preferences, cond= "input_received")
        | ask_foodType.to(process_preferences, cond= "input_received")
        
        | ask_qualifier.to(preference_reasoning, cond="qualifier_stated")
        | ask_qualifier.to(process_preferences,cond= "input_received")

        | state_preferences.to(completed, cond="exit_conversation")
        | state_preferences.to(state_preferences)
                
        | return_restaurant.to(give_information, cond="valid_request")
        | return_restaurant.to(process_alternative, cond="preference_change")
        | return_restaurant.to(completed, cond="exit_conversation")
        | return_restaurant.to(return_restaurant)

        | give_information.to(completed, cond="exit_conversation")
        | give_information.to(give_information, cond="valid_request")
        | give_information.to(process_alternative, cond="preference_change")
        | give_information.to(cant_give_information)
        
        | no_restaurant_found.to(completed, cond="exit_conversation")
        | no_restaurant_found.to(process_alternative,cond="preference_change")
        | no_restaurant_found.to(no_restaurant_found)
    )

    evaluate_input = (
        process_preferences.to(return_restaurant, cond="variables_known")
        | process_preferences.to(ask_levenshtein, unless="levenshtein_known")
        | process_preferences.to(ask_area, unless = "area_known")
        | process_preferences.to(ask_foodType, unless = "foodType_known")
        | process_preferences.to(ask_priceRange, unless = "priceRange_known")
        | process_preferences.to(ask_qualifier, unless="qualifier_known", cond="restaurants_left")
        | process_preferences.to(state_preferences)
    )

    no_restaurant_trans=(
        return_restaurant.to(no_restaurant_found)
    )
    many_restaurant_trans=(
        preference_reasoning.to(return_restaurant)
    )

    request_alternative=(
        process_alternative.to(ask_qualifier,cond="restaurants_left",unless="qualifier_known")
        | process_alternative.to(preference_reasoning,cond="qualifier_known")
        | process_alternative.to(return_restaurant)
    )

    information_trans=(
        cant_give_information.to(give_information)
    )

    # INITIATION
    def __init__(self,restaurant_file: str, classifier_file: Union[ClassifierMixin, KeywordClassifier], vectorizer_file: CountVectorizer,reasoning_file: str) -> None:
        """Initializes a RestaurantAgent instance.

        Args:
            restaurant_file (str): String pointing to the restaurant file
            classifier_file (Union[ClassifierMixin, KeywordClassifier]): The classifier to be used by the parser
            vectorizer_file (CountVectorizer): The trained CountVectorizer instance to be used.
        """
        self.area = ""
        self.foodType = ""
        self.priceRange = ""
        self.qualifier=""
        self.informationGiven = True #Gets passed to the give information setting to trigger the thinking scrips
        self.context = None #Gets passed to the classifier to help understand otherwise ambiguous answers
        self.tries = 0 #keep track of how many restaurants of the same variable combination were returned
        self.current_input = None #parsed current input so parsing only runs once per input
        self.stage = 0 #For the Levenshtein State to fill in the correct preference
        self.all_restaurants = pd.read_csv(restaurant_file)
        self.reasoning_rules = json.loads(open(reasoning_file,"rb").read())
        self.filteredRestaurants = None
        self.parser = TextParser(classifier_file,restaurant_file,vectorizer_file)
        self.current_suggestion = None
        self.current_suggestion_set = False
        self.no_res_passes = 0 #Changes the error message to be more helpful on repeated state entry.
        self.add_preferences = False #keeps track of addeditional preferences
        self.add_description = False
        self.turns = 0 #Keeps track of number of user prompts 
        self.preRestReturnUtterances = ["I think i might have just the place for you.", "Let me see what we have here", "Let me take a look", "So many places to choose from", "I' thinking about a place that could suit you"] #List of possible utterances
        self.preInfoUtterances = ["Let me look that up", "Just a sec", "I'll take a look for you", "Let me get that", "I'm taking a look"]
        super(RestaurantAgent, self).__init__(rtc=False)
    
    # HELPER FUNCTIONS
    def getTurns(self):
        return self.turns

    def processVariableDict(self,input: dict) -> None:
        """Helper function to assign variable ms from parsed data:

        Args:
            input (dict): The parsed data from the classifier
        """
        #print("processing input variables", input, self.current_input)
        classAnswer = self.current_input
        if classAnswer[0] == "affirm":
            self.levenshtein = False
        if len(classAnswer)==2:
            if(classAnswer[0] in ["inform","reqalts","confirm","negate","request"]):
                if ALLOW_MULTIPLE_PREFERENCES_PER_UTTERANCE:
                        #print(self.area, self.foodType,self.priceRange)
                        for key,val in classAnswer[1].items():
                            if key=="area":
                                self.area=val
                            elif key == "foodType" and (RANDOMIZE_PREFERENCE_QUESTION_ORDER or self.area!=""):
                                    self.foodType=val
                            elif key == "priceRange" and (RANDOMIZE_PREFERENCE_QUESTION_ORDER or (self.foodType !="" and self.area!="")):
                                    self.priceRange=val
                            elif key == "qualifier" and (RANDOMIZE_PREFERENCE_QUESTION_ORDER or (self.foodType !="" and self.area!="")):
                                self.qualifier=val
                else:
                    if len(classAnswer[1]) == 1:
                        key,val = classAnswer[1].items()
                        if key=="area":
                                self.area=val
                        elif key == "foodType" and (RANDOMIZE_PREFERENCE_QUESTION_ORDER or self.area!=""):
                                self.foodType=val
                        elif key == "priceRange" and (RANDOMIZE_PREFERENCE_QUESTION_ORDER or (self.foodType !="" and self.area!="")):
                                self.priceRange=val
                        elif key == "qualifier" and (RANDOMIZE_PREFERENCE_QUESTION_ORDER or (self.foodType !="" and self.area!="")):
                                self.qualifier=val
                    else:
                        chatbot_print("Sorry, we can only process one preference each sentence")

    def search_restaurant(self) -> pd.DataFrame:
        """Searches the restaurant database for restaurants matching the current variables

        Returns:
            pd.DataFrame: A dataframe containing the restaurants matching the current variables
        """
        df = self.all_restaurants
        if(self.area!="dontcare"):
            df = df[df["area"]==self.area]
        if(self.priceRange!="dontcare"):
            df=df[df["pricerange"]==self.priceRange]
        if(self.foodType!="dontcare"):
            df=df[df["food"]==self.foodType]
        return df

    #
    def no_response_formatter(self,other: bool=False) -> str:
        """This is to only mention specified information in the response.
        
        Returns: 
            str: The response
        """
        foodpart = f'serving {self.foodType} food' if (self.foodType!="" and self.foodType!="dontcare") else ""
        areapart = f'in the {self.area}' if (self.area!="" and self.area!="dontcare") else ""
        pricepart = f'that has {self.priceRange} prices' if (self.priceRange!="" and self.priceRange!="dontcare") else ""
        
        qualifierDict = {"romantic":"that is romantic","children":"thats child appropriate","assigned":"that has assigned seats","touristic":"that is touristic"}
        qualifierpart = f'{qualifierDict.get(self.qualifier,None)}' if (self.qualifier!="") else ""
        otherp = 'other ' if(other) else '' 
        resp = f"I'm sorry but there is no {otherp}restaurant {foodpart} {areapart} {pricepart},{qualifierpart}"
        return resp
        
    #CONDITIONAL TRANSITIONS
    def input_received(self, input: str) -> bool:
        """Checks whether the input is not None"""
        #print("Received: ", input)
        return input!=None
    
    def variables_known(self) -> bool:
        """Checks whether all variables are known
        """
        if(ASK_CONFIRMATION_LEVENSHTEIN):
            if(self.restaurants_left()):
                return self.area != "" and self.foodType!="" and self.priceRange != "" and self.levenshtein != True and self.qualifier != ""
            else:
                return self.area != "" and self.foodType!="" and self.priceRange != "" and self.levenshtein != True

        else:
            if(self.restaurants_left()):
                return self.area != "" and self.foodType!="" and self.priceRange != "" and self.qualifier != ""
            else:
                return self.area != "" and self.foodType!="" and self.priceRange != ""
    
    def restaurants_left(self)->bool:
        if(len(self.search_restaurant())>1):
            return True
        else: 
            return False

    def negate_or_thanks(self)->bool:
        if(self.current_input[0] in ["thankyou","negate"]):
            return True
        elif(self.current_input[1]=={}):
            True
        else:
            return False
    
    def qualifier_stated(self)->bool:
        if(len(self.current_input)>1):
            var = self.current_input[1].get("qualifier")
            if(var!=None and var!=""):
                self.qualifier=var
                return True
        self.qualifier="None"
        return False

    def levenshtein_known(self) -> bool:
        """Returns a bool representing whether program use the levenshtein
        """
        if(ASK_CONFIRMATION_LEVENSHTEIN):
            return not self.levenshtein
        else:
            return True

    def area_known(self) -> bool:
        """Returns a bool representing whether the area is known"""
        return self.area!=""
    
    def priceRange_known(self) -> bool:
        """Returns a bool representing whether the price range is known"""
        return self.priceRange!=""
    
    def foodType_known(self) -> bool:
        """Returns a bool representing whether the food type is known"""
        return self.foodType!=""
    
    def qualifier_known(self) -> bool:
        """Returns a bool representing whether the qualifier is known"""
        return self.qualifier!=""

    def valid_request(self,input: str) -> bool:
        """Checks whether the input is a valid request for information about the current restaurant"""
        input_type = self.current_input[0]
        #print("valid request", input,self.current_input)
        return input_type=="request" and self.current_suggestion_set and not self.preference_change(input)
    
    def exit_conversation(self, input: str) -> bool:
        """Checks whether the input is a valid exit"""
        exit_input = self.current_input[0]
        #print("valid exit", input,self.current_input)
        return exit_input=="bye" or exit_input=="thankyou"
    
    def preference_change(self, input: str) -> bool:
        """Checks whether the input is a valid preference change"""
        #print("preferences changed?", input,self.current_input)
        if(len(self.current_input)>1):
            return (type(self.current_input[1])==dict and len(self.current_input[1]) > 0)
        else:
            return False
    
    #EXIT FUNCTIONS
    def on_exit_state_preferences(self, input: str) -> None:
        """Runs when the user exits the state_preferences state"""
        self.processVariableDict(input)

    def on_exit_ask_levenshtein(self, input: str) -> None:
        """Runs when the user exits the ask_levenshtein state"""
        self.processVariableDict(input)
    def on_exit_ask_area(self, input: str) -> None:
        """Runs when the user exits the ask_area state"""
        self.processVariableDict(input)
        
    def on_exit_ask_priceRange(self, input: str) -> None:
        """Runs when the user exits the ask_priceRange state"""
        self.processVariableDict(input)
        
    def on_exit_ask_foodType(self, input: str) -> None:
        """Runs when the user exits the ask_foodType state"""
        self.processVariableDict(input)
    
    #ENTRY FUNCTIONS
    def on_enter_ask_levenshtein(self) -> None:
        """Runs when the user enters the ask_levenshtein state"""
        str = "Let me confirm. Are you searching for the restaurants with "
        if self.stage == 0:
            if self.area != "":
                str += f"the area of {self.area}"
            else:
                str += "no area"
            if self.priceRange != "":
                str += f", the price range of {self.priceRange}"
            else:
                str += ", no price range"
            if self.foodType != "":
                str += f" and the food type of {self.foodType}?"
            else:
                str += " and no food type?"
        elif self.stage == 1:
            if self.area != "":
                str += f"the area of {self.area}?"
            else:
                str += "no area?"
        elif self.stage == 2:
            if self.priceRange != "":
                str += f"the price range of {self.priceRange}?"
            else:
                str += "no price range?"
        elif self.stage == 3:
            if self.foodType != "":
                str += f"the food type of {self.foodType}?"
            else:
                str += "no food type?"
        chatbot_print(str)

    def on_enter_ask_area(self) -> None:
        """Runs when the user enters the ask_area state"""
        self.stage = 1
        self.turns+=1
        chatbot_print("What part of town do you have in mind?")
        self.context="area"

    def on_enter_ask_priceRange(self) -> None:
        self.stage = 2
        """Runs when the user enters the ask_priceRange state"""
        self.turns+=1
        chatbot_print("Would you like something in the cheap , moderate , or expensive price range?")
        self.context="priceRange"
    
    def on_enter_ask_foodType(self) -> None:
        self.stage = 3
        """Runs when the user enters the ask_foodType state"""
        self.turns+=1        
        chatbot_print("What kind of food would you like?")
        self.context="foodType"
    
    def on_enter_ask_qualifier(self) -> None:
        """Runs when the user enters the ask_qualifier state"""
        self.turns+=1        
        chatbot_print("Do you have additional requirements?")

    def on_enter_hello(self) -> None:
        """Runs when the user enters the hello state"""
        self.turns+=1        
        chatbot_print("Hello , welcome to the UU restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
        self.context=None
        self.send("start_processing")
    
    def on_enter_cant_give_information(self) -> None:
        """Runs when the user enters the cant_give_information state"""
        self.informationGiven = False
        self.turns+=1
        chatbot_print("I'm sorry i did not understand your request")
        self.context=None
        self.send("information_trans",input="")
        
    def on_enter_no_restaurant_found(self) -> None:
        """Runs when the user enters the no_restaurant_found state"""
        if(self.no_res_passes>0):
            self.turns+=1
            chatbot_print("Sorry, but there are no such restaurants, maybe try changing the location, area or foodtype?")
            self.context=None
        else:
            resp = self.no_response_formatter(other=(self.tries>0))
            self.turns+=1
            chatbot_print(resp)
            self.context=None
        self.no_res_passes+=1
        self.current_suggestion = None
        self.filteredRestaurants=None
        self.current_suggestion_set = False

    
    def on_enter_preference_reasoning(self)->None:
        self.add_preferences=True
        self.reasoning_rules
        neg_cond = self.reasoning_rules[self.qualifier+"-false"]
        pos_cond = self.reasoning_rules[self.qualifier+"-true"]
        use_description = ""
        df = self.search_restaurant()
        pdf = self.filteredRestaurants #Dataframe with positive adding qualifiers
        ndf = self.filteredRestaurants #Dataframe with non-negative qualifiers
        for cond in pos_cond["conditions"]:
            pdf=df[df[cond[0]]==cond[1]]
        for cond in neg_cond["conditions"]:
            ndf=df[df[cond[0]]!=cond[1]]
        first_df = pd.merge(pdf, ndf, how ='inner')
        #restaurant with non-negative attributes come second
        second_df = ndf.merge(first_df, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
        #restaurant with both positive and negative attributes come third
        third_df = pdf.merge(first_df, how='left', indicator=True).query('_merge == "left_only"').drop(columns=['_merge'])
        first_df["reason"]=pos_cond["description"]
        second_df["reason"] = neg_cond["description"]
        third_df["reason"] = pos_cond["description"]
        res_df = pd.concat([first_df, second_df, third_df], ignore_index=True)
        
        self.filteredRestaurants = res_df
        self.add_description=True
        self.tries = 0
        self.send("many_restaurant_trans")
            
        
                

    def on_enter_return_restaurant(self) -> None:
        """Runs when the user enters the return_restaurant state"""
        if(self.filteredRestaurants is None):
            self.filteredRestaurants = self.search_restaurant()
        if(CHOSEN_SYSTEM=="A"):
            chatbot_print(random.choice(self.preRestReturnUtterances))
        if(len(self.filteredRestaurants)<self.tries+1):
            self.send("no_restaurant_trans")
        else:
            row = self.filteredRestaurants.iloc[self.tries]
            self.current_suggestion = row
            self.current_suggestion_set = True
            self.tries+=1
            self.no_res_passes=0
            self.context=None
            chatbot_print(f"{row['restaurantname']} is a nice place in the {row['area']} part of town serving {row['food']} food and the prices are {row['pricerange']}")
            if(self.add_description):
                chatbot_print(f"{row['restaurantname']} was chosen because "+row['reason'])
    
    def on_enter_give_information(self, input: str) -> None:
        """There are three possibilities: 
            the user can request an alternative updating the variables, 
            the user can request specific information about the current restaurant, 
            or the user can ask for information about the current restaurant without specifying what they want to know.
        """
        if(self.current_suggestion_set):
            if(CHOSEN_SYSTEM=="A" and self.informationGiven):
                chatbot_print(random.choice(self.preInfoUtterances))
            self.informationGiven = True
            self.context=None
            request_type,_ = self.parser.parseText(input, requestPossible=True)
            response_dict = {
                "phone": "phone number",
                "postcode": "postcode",
                "addr": "address",
                "food": "food served",
                "pricerange": "price range",
                "restaurantname": "name",
                "area": "area"
            }
            response = ""
            for key in request_type[1]:
                if key in response_dict:
                    attribute = response_dict[key]
                    value = self.current_suggestion[key]
                    response += f"the {attribute} of {self.current_suggestion['restaurantname']} is {value}, " if not response else f"their {attribute} is {value}, "
            
            if response != "":
                # Format and #print the bot answer
                chatbot_print(response.capitalize()[:-2] + ".")
                return
            self.turns+=1
            chatbot_print("Can you provide specific information you are looking for such as phone number, area or address?")
            
            
    def on_enter_completed(self) -> None:
        """Runs when the user exits the system"""
        chatbot_print("Thank you for using the UU restaurant system. Goodbye!")
    
    def on_enter_process_alternative(self, input: str) -> None:
        """Processes the input and updates the variables accordingly"""
        self.qualifier=""
        self.processVariableDict(input)
        self.filteredRestaurants = None
        self.current_suggestion = None
        self.current_suggestion_set = False
        self.tries = 0
        self.no_res_passes = 0
        self.add_preferences = False
        self.add_description = False
        
        #print("requesting with updated variables")
        self.send("request_alternative") #Auto transition to return restaurant.
    
    def on_enter_process_preferences(self, input:str) -> None:
        """Runs when the user enters the process_preferences state"""
        #print("Entering process preferences")
        self.send("evaluate_input")


    # INPUT HANDLING
    def input_step(self, user_input: str) -> str:
        """Parses the input and sends it to the state machine"""
        #print(self.current_state)
        input, self.levenshtein = self.parser.parseText(user_input,context=self.context,requestPossible=False)
        #wprint(input)
        self.current_input = input
        #print("Classifier output",input,"from: ",user_input)
        self.send("receive_input", input=user_input)
        #print(self.current_state)
        
        
    
    def graph(self, filename: str="") -> DotGraphMachine:
        """Save a graph of the state machine with the current state highlighted to specified file"""
        #return
        diagram_graph = DotGraphMachine(self)
        if (filename != ""):
            diagram_graph().write_png(filename)
        return diagram_graph


def main() -> None:
    # Set up state machine
    clf_data = "data/dialog_acts.dat"
    data = prepare_data(clf_data)
    classifier = train_model(data["complete"], "DecisionTree")
    vectorizer = data["complete"]["vectorizer"]
    reasoning_file = "data/reasoning_rules.json"
    restaurant_file = "data/restaurant_info.csv"
    sm = RestaurantAgent(restaurant_file,classifier,vectorizer,reasoning_file)
    
    #sm.graph("initial.png")
    """
    ##print(sm.current_state)
    sm.send("start_processing")

    # Test inputs
    
    sm.input_step("im looking for an expensive restaurant that serves european food")
    sm.input_step("any")
    sm.input_step("anything else")
    sm.input_step("whats the address")
    sm.input_step("what area is it in")
    sm.input_step("how about italian food")
    #print(sm.completed.is_active)
    #sm.input_step("Goodbye")

    #print(sm.completed.is_active)
    """
    testfile = open("test.txt","rb")
    cont=False #continues for one round after the loop stops to allow for restart through the testfile.
    start_time = time.time()
    while not sm.completed.is_active or (cont):
        nxtline = testfile.readline().decode().strip()
        if(DIALOG_TESTING and nxtline!=""):
            if(nxtline=="#"):
                #print("resetting testing agent")
                sm = RestaurantAgent(restaurant_file,classifier,vectorizer,reasoning_file)
                cont=True
            else:
                #print("Auto Input: ",nxtline)F
                sm.input_step(nxtline)
                cont=True
        else:
            user_input = take_user_input()
            sm.input_step(user_input)
            cont=False
    end_time = time.time()
    elapsed_time = end_time-start_time
    print(f'System: {CHOSEN_SYSTEM}: Turns: {sm.getTurns()} Time: {elapsed_time}')
        

if __name__ == '__main__':
    main()
