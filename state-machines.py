from statemachine import StateMachine, State
import pandas as pd
import pickle as pkl
from textParser import TextParser
from statemachine.contrib.diagram import DotGraphMachine
import warnings
warnings.filterwarnings("ignore")

#Basic State machine that returns the number of restaurants in an area of town given the keyword south, west, east, north and center.
class RestaurantAgent(StateMachine):
    # STATES
    hello = State(initial=True)
    waiting_for_input = State()
    return_restaurant = State()
    process_input = State()
    give_information = State()
    completed = State(final=True)
    
    # TRANSITIONS
    receive_input = (
        waiting_for_input.to(process_input, cond="input_received")
        | waiting_for_input.to(waiting_for_input, unless="input_received")
    )
    evaluate_input = (
        process_input.to(return_restaurant, cond="variables_known")
        | process_input.to(waiting_for_input, unless="variables_known")
    )
    start_processing = hello.to(waiting_for_input)
    
    provide_information = (
        return_restaurant.to(completed, cond="exit_conversation")
        | return_restaurant.to(give_information, unless="exit_conversation")
    )
    reprovide_information = (
        give_information.to(give_information)
    )
    
    complete_process = (give_information.to(completed, cond="exit_conversation")
                        | give_information.to(return_restaurant, unless="exit_conversation")
    )

    request_alternative = (return_restaurant.to(return_restaurant)
                        | give_information.to(return_restaurant)
    )

    
    def __init__(self,restaurant_file, classifier_file, vectorizer_file) -> None:
        self.counter = 0 #keep track of number of times initial input state was entered
        self.area = "" #if area is valid, area is stored here
        self.foodType = ""
        self.priceRange = ""
        self.context = None
        self.tries = 0 #keep track of how many restaurants of the same variable combination were returned
        self.all_restaurants = pd.read_csv(restaurant_file)
        self.filteredRestaurants = None
        self.parser = TextParser(classifier_file,restaurant_file,vectorizer_file)
        self.current_suggestion = None
        self.current_suggestion_set = False
        super(RestaurantAgent, self).__init__(rtc=False)
    
    # HELPER FUNCTIONS
    #Helper function to assign variables:
    def processVariableDict(self,variables) -> None:
        """Helper function to assign variables"""
        for key,val in variables.items():
            if key == "foodType":
                    self.foodType=val
            elif key == "priceRange":
                    self.priceRange=val
            elif key == "area":
                    self.area=val
    
    def search_restaurant(self) -> pd.DataFrame:
        """Searches for a restaurant based on the current variables"""
        df = self.all_restaurants
        if(self.area!="dontcare"):
            df = df[df["area"]==self.area]
        if(self.priceRange!="dontcare"):
            df=df[df["pricerange"]==self.priceRange]
        if(self.foodType!="dontcare"):
            df=df[df["food"]==self.foodType]
        return df

    #This is to only mention specified information in the response.
    def no_response_formatter(self,other: bool=False) -> str:
        """Generates a response when no restaurants are found"""
        foodpart = f'serving {self.foodType} food' if (self.foodType!="" and self.foodType!="dontcare") else ""
        areapart = f'in the {self.area}' if (self.area!="" and self.area!="dontcare") else ""
        pricepart = f'that has {self.priceRange} prices' if (self.priceRange!="" and self.priceRange!="dontcare") else ""
        otherp = 'other' if(other) else '' 
        resp = f"I'm sorry but there is no {otherp} restaurant {foodpart} {areapart} {pricepart}"
        return resp
        
    #CONDITIONAL TRANSITIONS
    def input_received(self, input: str) -> bool:
        return input!=None
    
    def variables_known(self) -> bool:
        return self.area != "" and self.foodType!="" and self.priceRange != ""
    
    def exit_conversation(self, input: str) -> bool:
        exit_input = self.parser.parseText(input)[0]
        return exit_input=="bye" or exit_input=="thankyou"
    
    def preference_change(self, input: str) -> bool:
        return (len(self.parser.parseText(input)[1]) != 0)
    
    #ENTRY & EXIT FUNCTIONS
    def on_exit_waiting_for_input(self, input) -> None:
        """Processes the user's input"""
        classAnswer = self.parser.parseText(input,context=self.context,requestPossible=False)
        if len(classAnswer)==2: 
            if(classAnswer[0] in ["inform","reqalts","confirm","negate","request"]):
                self.processVariableDict(classAnswer[1])

    def on_enter_waiting_for_input(self) -> None:
        """ Sets the current context"""
        if(self.counter>0):
            if(self.area==""):
                print("What part of town do you have in mind?")
                self.context="area"
            elif(self.priceRange==""):
                print("Would you like something in the cheap , moderate , or expensive price range?")
                self.context="priceRange"
            elif(self.foodType==""):
                print("What kind of food would you like?")
                self.context="foodType"
            else:
                self.context = None
                print("I did not understand your last input, can we try again?")
        self.counter+=1
        

    def on_enter_hello(self) -> None:
        print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
        self.send("start_processing")

    
    def on_enter_return_restaurant(self) -> None:
        """Generates an answer when the user requests a restaurant"""
        if(self.filteredRestaurants is None):
            self.filteredRestaurants = self.search_restaurant()
            if (len(self.filteredRestaurants) == 0):
                resp = self.no_response_formatter()
                print(resp)
                self.current_suggestion = None
                self.filteredRestaurants=None
                self.current_suggestion_set = False
                return
            row = self.filteredRestaurants.iloc[0]
            self.current_suggestion = row
            self.current_suggestion_set = True
            self.tries=1
        else:
            if(len(self.filteredRestaurants) <= self.tries + 1):
                resp = self.no_response_formatter(other="true")
                print(resp)
                self.current_suggestion = None
                self.filteredRestaurants=None
                self.current_suggestion_set = False
                return
            row = self.filteredRestaurants.iloc[self.tries]
            self.current_suggestion = row
            self.current_suggestion_set = True
            self.tries+=1
        print(f"{row['restaurantname']} is a nice place in the {row['area']} part of town serving {row['food']} food and the prices are {row['pricerange']}")
        
        
    def on_enter_give_information(self, input: str) -> None:
        """Generates an answer when the user requests information about a restaurant"""
        requestPossible = self.current_suggestion_set #Only enable information requests if a restaurant is loaded
        request_type = self.parser.parseText(input, context=self.context, requestPossible=requestPossible)

        if(self.current_suggestion_set):
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
                # Format and print the bot answer
                print(response.capitalize()[:-2] + ".")
                return
            print("Can you provide specific information you are looking for such as phone number, area or address?")
        else:
            if(type(request_type[1])==dict):
                self.send("request_alternative", input=input)
                return
            print("Sorry, can you try changing the area, price or foodtype?")
            
            
    def on_enter_completed(self) -> None:
        """Runs when the user exits the system"""
        print("Thank you for using the UU restaurant system. Goodbye!")


    def on_request_alternative(self,input) -> None:
        """Finds an alternative restaurant when the user requests one"""
        if (self.preference_change(input)):
            preferences = self.parser.parseText(input,requestPossible=False)[1]
            self.processVariableDict(preferences)
            self.filteredRestaurants = None
            self.tries = 0


    # INPUT HANDLING
    def input_step(self, user_input: str) -> str:
        """Represents a single step of the input loop"""
        input = self.parser.parseText(user_input) #["inform","reqalts","confirm","negate","request"]
        if self.current_state.id == "waiting_for_input":
            self.send("receive_input", input=user_input)
            self.send("evaluate_input")
        elif self.current_state.id == "return_restaurant":
            if (input[0] in ["inform","reqalts","negate","confirm"] or (not self.current_suggestion_set and input[0]=="request")):
                self.send("request_alternative", input=user_input)
            else:
                self.send("provide_information", input=user_input)
        elif self.current_state.id == "give_information":
            if (input[0] in ["inform","reqalts","negate","confirm"]):
                self.send("request_alternative", input=user_input)
            elif (input[0] in ["request"]):
                self.send("reprovide_information", input=user_input)
            else:
                self.send("complete_process", input=user_input)
        
    
    def graph(self, filename: str="") -> DotGraphMachine:
        """
        Save a graph of the state machine with the current state highlighted to specified file
        """
        graph = DotGraphMachine(self)
        if (filename != ""):
            graph().write_png(filename)
        return graph


def main() -> None:
    # Set up state machine
    restaurant_file = "restaurant_info.csv"
    classifier = pkl.load(open("./ass_1a/models/complete/DecisionTree.pkl",'rb'))
    vectorizer = pkl.load(open("./ass_1a/models/complete/vectorizer.pkl",'rb'))
    sm = RestaurantAgent(restaurant_file,classifier,vectorizer)

    # User input loop
    while not sm.completed.is_active:
        user_input = input("Type your response: ")
        sm.input_step(user_input)
        
        

if __name__ == '__main__':
    main()
