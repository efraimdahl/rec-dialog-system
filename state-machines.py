from statemachine import StateMachine, State
import pandas as pd
import pickle as pkl
from textParser import TextParser

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
    
    complete_process = (give_information.to(completed, cond="exit_conversation")
                        | give_information.to(return_restaurant, unless="exit_conversation")
    )

    
    def __init__(self,restaurant_file,classifier_file,vectorizer_file):
        self.counter = 0 #keep track of number of times initial input state was entered
        self.area = "" #if area is valid, area is stored here
        self.foodType = ""
        self.priceRange = ""
        self.context = None
        self.tries = 0 #keep track of how many restaurants of the same variable combination where returned
        self.all_restaurants = pd.read_csv(restaurant_file)
        self.filteredRestaurants = None
        self.parser = TextParser(classifier_file,restaurant_file,vectorizer_file)
        super(RestaurantAgent, self).__init__(rtc=False)
    
    # HELPER FUNCTIONS
    #Helper function to assign variables:
    def processVariableDict(self,variables):
        for key,val in variables.items():
            match key:
                case "foodType":
                    self.foodType=val
                case "priceRange":
                    self.priceRange=val
                case "area":
                    self.area=val
    
    def search_restaurant(self):
        df = self.all_restaurants
        if(self.area!="dontcare"):
            df = df[df["area"]==self.area]
        if(self.priceRange!="dontcare"):
            df=df[df["pricerange"]==self.priceRange]
        if(self.foodType!=["dontcare"]):
            df=df[df["food"]==self.foodType]
        return df
       
        
    #CONDITIONAL TRANSITIONS
    def input_received(self,input):
        return input!=None
    
    def variables_known(self):
        #print("checking for variables",self.area, self.priceRange, self.foodType)
        return self.area != "" and self.foodType!="" and self.priceRange != ""
    
    def exit_conversation(self, input):
        return self.parser.parseText(input)[0]=="bye"
    
    #ENTRY & EXIT FUNCTIONS
    def on_exit_waiting_for_input(self,input):
        classAnswer = self.parser.parseText(input,context=self.context,requestPossible=False)
        print(input,classAnswer,self.context)
        if len(classAnswer)==2: 
            if(classAnswer[0] in ["inform","reqalts","confirm","negate","request"]):
                self.processVariableDict(classAnswer[1])
    
    def on_enter_waiting_for_input(self):
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
        

    def on_enter_hello(self):
        print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
        
    def on_enter_provide_information(self, input):
        request_type = self.parser.parseText(input,context=self.context,requestPossible=True)
        print(input, request_type)
    
    def on_enter_return_restaurant(self):
        print(self.filteredRestaurants)
        if(self.filteredRestaurants == None):
            self.filteredRestaurants = self.search_restaurant()
            row = self.filteredRestaurants.iloc[0]
            self.current_suggestion = row
            self.tries=1
        else:
            row = self.filteredRestaurants.iloc[i]
            self.current_suggestion = row
            self.tries+=1
        print(f"{row['restaurantname']} is a nice place in the {row['area']} part of town serving {row['food']} food and the prices are {row['pricerange']}")
        
    def on_enter_give_information(self, input):
        request_type = self.parser.parseText(input, context=self.context, requestPossible=True)
        
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
        
        if response:
            print(response.capitalize()[:-2] + ".")
            
    def on_enter_complete_process(self):
        print("Thank you for using the Cambridge restaurant system. Goodbye!")

    # INPUT HANDLING
    def input_step(self, user_input: str) -> str:
        
        print(f"Before state: {self.current_state}")
        print(f"User message: {user_input}")
        print(self.parser.parseText(user_input))
        if self.current_state.id == "waiting_for_input":
            self.send("receive_input", input=user_input)
            self.send("evaluate_input")
        elif self.current_state.id == "return_restaurant":
            self.send("provide_information", input=user_input)
        elif self.current_state.id == "give_information":
            self.send("complete_process", input=user_input)
        
        
        print(f"After state: {self.current_state}")
        
    

def main():
    restaurant_file = "restaurant_info.csv"
    classifier_file = "./ass_1a/models/complete/Ridge.pkl"
    vectorizer_file = "./ass_1a/models/complete/vectorizer.pkl"

    restaurant_file = "restaurant_info.csv"
    sm = RestaurantAgent(restaurant_file,classifier_file,vectorizer_file)
    sm.send("start_processing")

    # Test inputs
    sm.input_step("I am looking for an italian restaurant in the centre of town")
    sm.input_step("expensive")
    sm.input_step("What is their phone number and address?")
    
    sm.input_step("Goodbye")
    # TODO When you change the last msg to another dialogue act,
    # The program should request whether preferences have changed. If so, go back to getting the preferences.
    # If not, return the next restaurant in the dataframe. If there is none, tell the user that.
    
    # User input loop
    #while True:
    #    user_input = input("Type your response: ")
    #    sm.input_step(user_input)
        


if __name__ == '__main__':
    main()


