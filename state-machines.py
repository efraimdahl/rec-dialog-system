from statemachine import StateMachine, State
import pandas as pd
import pickle as pkl
from textParser import TextParser

#Basic State machine that returns the number of restaurants in an area of town given the keyword south, west, east, north and center.
class RestaurantAgent(StateMachine):
    
    #STATES
    hello = State(initial=True)
    waiting_for_input = State()
    return_restaurant = State()
    process_input = State()
    completed = State(final=True)
    
    #TRANSITIONS
    receive_input = (
        waiting_for_input.to(process_input, cond="input_received")
        | waiting_for_input.to(waiting_for_input, unless="input_received")
    )
    evaluate_input=(
        process_input.to(return_restaurant, cond="variables_known")
        | process_input.to(waiting_for_input, unless="variables_known")
    )
    start_processing = hello.to(waiting_for_input)
    complete_process = return_restaurant.to(completed)
    
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
    
    #HELPER FUNCTIONS
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
            df=df[df["pricerange"]==self.pricerange]
        if(self.foodType!=["dontcare"]):
            df=df[df["food"]==self.foodType]
        return df
       
        
    #CONDITIONAL TRANSITIONS
    def input_received(self,input):
        return input!=None
    
    def variables_known(self):
        #print("checking for variables",self.area, self.priceRange, self.foodType)
        return self.area != "" and self.foodType!="" and self.priceRange != ""
    
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
    
    def on_enter_return_restaurant(self):
        if(self.filteredRestaurants==None):
            self.filteredRestaurants = self.search_restaurant()
            row = self.filteredRestaurants.iloc[0]
            self.tries=1
        else:
            row = self.filteredRestaurants.iloc[i]
            self.tries+=1
        print(f"{row['restaurantname']} is a nice place in the {row['area']} part of town serving {row['food']} food and the prices are {row['pricerange']}")


def main():
    restaurant_file = "restaurant_info.csv"
    classifier_file = "./ass_1a/models/complete/Ridge.pkl"
    vectorizer_file = "./ass_1a/models/complete/vectorizer.pkl"

    restaurant_file = "restaurant_info.csv"
    sm = RestaurantAgent(restaurant_file,classifier_file,vectorizer_file)
    print(sm.current_state)
    sm.send("start_processing")
    print(sm.current_state)
    sm.send("receive_input",input="I am looking for an italian restaurant in the centre of town")
    print(sm.current_state)
    sm.send("evaluate_input")
    print(sm.current_state)
    sm.send("receive_input",input="any price")
    print(sm.current_state)
    sm.send("evaluate_input")
    print(sm.current_state)


if __name__ == '__main__':
    main()


