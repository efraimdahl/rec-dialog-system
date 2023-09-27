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
    state_preferences = State()
    process_preferences = State()
    ask_area = State()
    ask_priceRange = State()
    ask_foodType = State()
    return_restaurant = State()
    no_restaurant_found = State()
    process_alternative = State()
    give_information = State()
    completed = State(final=True)
    
    # 
    start_processing = hello.to(state_preferences)
    
    receive_input = (
        state_preferences.to(process_preferences, cond="input_received")
        | ask_area.to(process_preferences, cond= "input_received")
        | ask_priceRange.to(process_preferences, cond= "input_received")
        | ask_foodType.to(process_preferences, cond= "input_received")
        | state_preferences.to(completed, cond="exit_conversation")
        | state_preferences.to(state_preferences, unless="input_received")
        
        | return_restaurant.to(process_alternative, cond="preference_change")
        | return_restaurant.to(give_information, cond="valid_request")
        | return_restaurant.to(completed, cond="exit_conversation")
        | return_restaurant.to(return_restaurant, unless="valid_request")

        | give_information.to(process_alternative, cond="preference_change")
        | give_information.to(give_information, cond="valid_request")
        | give_information.to(completed, cond="exit_conversation")

        | no_restaurant_found.to(process_alternative,cond="preference_change")
        | no_restaurant_found.to(completed, cond="exit_conversation")
        | no_restaurant_found.to(no_restaurant_found, unless=["preference_change","exit_conversation"])

    )

    evaluate_input = (
        process_preferences.to(return_restaurant, cond="variables_known")
        | process_preferences.to(ask_area, unless = "area_known")
        | process_preferences.to(ask_foodType, unless = "foodType_known")
        | process_preferences.to(ask_priceRange, unless = "priceRange_known")
        | process_preferences.to(state_preferences)
        | process_alternative.to(return_restaurant, cond="variables_known")
        | process_alternative.to(state_preferences)
    )

    no_restaurant_trans=(
        return_restaurant.to(no_restaurant_found)
    )


    def __init__(self,restaurant_file,classifier_file,vectorizer_file):
        self.area = ""
        self.foodType = ""
        self.priceRange = ""
        self.context = None
        self.tries = 0 #keep track of how many restaurants of the same variable combination were returned
        self.current_input = None #parsed current input so parsing only runs once per input
        self.all_restaurants = pd.read_csv(restaurant_file)
        self.filteredRestaurants = None
        self.parser = TextParser(classifier_file,restaurant_file,vectorizer_file)
        self.current_suggestion = None
        self.current_suggestion_set = False
        super(RestaurantAgent, self).__init__(rtc=False)
    
    # HELPER FUNCTIONS
    #Helper function to assign variables from parsed data:
    def processVariableDict(self,input):
        print("processing input variables",input, self.current_input)
        classAnswer = self.current_input
        if len(classAnswer)==2: 
            if(classAnswer[0] in ["inform","reqalts","confirm","negate","request"]):
                for key,val in classAnswer[1].items():
                    if key == "foodType":
                            self.foodType=val
                    elif key == "priceRange":
                            self.priceRange=val
                    elif key == "area":
                            self.area=val
    
    def search_restaurant(self):
        df = self.all_restaurants
        if(self.area!="dontcare"):
            df = df[df["area"]==self.area]
        if(self.priceRange!="dontcare"):
            df=df[df["pricerange"]==self.priceRange]
        if(self.foodType!="dontcare"):
            df=df[df["food"]==self.foodType]
        return df

    #This is to only mention specified information in the response.
    def no_response_formater(self,other=False):
        foodpart = f'serving {self.foodType} food' if (self.foodType!="" and self.foodType!="dontcare") else ""
        areapart = f'in the {self.area}' if (self.area!="" and self.area!="dontcare") else ""
        pricepart = f'that has {self.priceRange} prices' if (self.priceRange!="" and self.priceRange!="dontcare") else ""
        otherp = 'other' if(other) else '' 
        resp = f"I'm sorry but there is no {otherp} restaurant {foodpart} {areapart} {pricepart}"
        return resp
        
    #CONDITIONAL TRANSITIONS
    def input_received(self,input):
        print("Received: ", input)
        return input!=None
    
    def variables_known(self):
        #print("checking for variables",self.area, self.priceRange, self.foodType)
        return self.area != "" and self.foodType!="" and self.priceRange != ""
    
    def area_known(self):return self.area!=""
    def priceRange_known(self):return self.priceRange!=""
    def foodType_known(self):return self.foodType!=""

    def valid_request(self,input):
        input_type = self.current_input[0]
        print("valid request", input,self.current_input)
        return input_type=="request" and self.current_suggestion_set and not self.preference_change(input)
    
    def exit_conversation(self, input):
        exit_input = self.current_input[0]
        print("valid exit", input,self.current_input)
        return exit_input=="bye" or exit_input=="thankyou"
    
    def preference_change(self,input):
        print("preferences changed", input,self.current_input)
        return (len(self.current_input[1]) > 0)
    
    #EXIT FUNCTIONS
    def on_exit_state_preferences(self, input):
        self.processVariableDict(input)
    def on_exit_ask_area(self, input):
        self.processVariableDict(input)
    def on_exit_ask_priceRange(self, input):
        self.processVariableDict(input)
    def on_exit_ask_foodType(self, input):
        self.processVariableDict(input)
    
    #ENTRY FUNCTIONS
    def on_enter_ask_area(self):
        print("What part of town do you have in mind?")
        self.context="area"    

    def on_enter_ask_priceRange(self):
        print("Would you like something in the cheap , moderate , or expensive price range?")
        self.context="priceRange"
    
    def on_enter_ask_foodType(self):
        print("What kind of food would you like?")
        self.context="foodType"

    def on_enter_hello(self):
        print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
        self.send("start_processing")
    
    def on_enter_no_restaurant_found(self):
        resp = self.no_response_formater(other=(self.tries>0))
        print(resp)
        self.current_suggestion = None
        self.filteredRestaurants=None
        self.current_suggestion_set = False

    def on_enter_return_restaurant(self):
        if(self.filteredRestaurants is None):
            self.filteredRestaurants = self.search_restaurant()
        if(len(self.filteredRestaurants)<self.tries+1):
            self.send("no_restaurant_trans")
        else:
            row = self.filteredRestaurants.iloc[self.tries]
            self.current_suggestion = row
            self.current_suggestion_set = True
            self.tries+=1
        print(f"{row['restaurantname']} is a nice place in the {row['area']} part of town serving {row['food']} food and the prices are {row['pricerange']}")
        
    #There are three possibilities, the user can request an alternative updatin
    def on_enter_give_information(self, input):
        if(self.current_suggestion_set):
            request_type = self.parser.parseText(input, context=self.context, requestPossible=True)
        else:
            request_type = self.current_input
        #print(request_type,self.current_suggestion)
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
            
            if response !="" : 
                print(response.capitalize()[:-2] + ".")
                return
            print("Can you provide specific information you are looking for such as phone number, area or address?")
        else:
            print("Sorry, can you try changing the area, price or foodtype?")
            
    def on_enter_completed(self):
        print("Thank you for using the UU restaurant system. Goodbye!")
    
    #STATE FUNCTIONS
    def on_process_alternative(self,input):
        self.processVariableDict(input,requestPossible=False)
        self.filteredRestaurants = None
        self.current_suggestion = None
        self.current_suggestion_set = False
        self.tries = 0
    
    def on_enter_process_preferences(self,input):
        print("Entering process preferences")
        self.send("evaluate_input")

    # INPUT HANDLING
    def input_step(self, user_input: str) -> str:
        print(self.current_state)
        input = self.parser.parseText(user_input)
        self.current_input = input
        print("Classifier output",input,"from: ",user_input)
        self.send("receive_input", input=user_input)
        print(self.current_state)
        
        
    
    def graph(self,filename=""):
        """
        Save a graph of the state machine with the current state highlighted to specified file
        """
        return
        diagram_graph = DotGraphMachine(self)
        if (filename != ""):
            diagram_graph().write_png(filename)
        return diagram_graph


def main():
    restaurant_file = "restaurant_info.csv"
    classifier = pkl.load(open("./ass_1a/models/complete/DecisionTree.pkl",'rb'))
    vectorizer = pkl.load(open("./ass_1a/models/complete/vectorizer.pkl",'rb'))
    restaurant_file = "restaurant_info.csv"
    sm = RestaurantAgent(restaurant_file,classifier,vectorizer)
    """
    sm.graph("initial.png")
    
    #print(sm.current_state)
    sm.send("start_processing")

    # Test inputs
    
    sm.input_step("im looking for an expensive restaurant that serves european food")
    sm.input_step("any")
    sm.input_step("anything else")
    sm.input_step("whats the address")
    sm.input_step("what area is it in")
    sm.input_step("how about italian food")
    print(sm.completed.is_active)
    #sm.input_step("Goodbye")

    print(sm.completed.is_active)
    """
    # User input loop
    while not sm.completed.is_active:
        user_input = input("Type your response: ")
        sm.input_step(user_input)
        


if __name__ == '__main__':
    main()
    
    


