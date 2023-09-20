



#Basic State machine that returns the number of restaurants in an area of town given the keyword south, west, east, north and center.

from statemachine import StateMachine, State
import pandas as pd
import pickle as pkl

#The textParser receives a trained classifier as input, and will return packets such as 
# 
#inform(type=restaurant,pricerange=moderate,task=find)
class TextParser():
    def __init__(self,classifier_file, restaurant_file):
        self.restaurant_data = pd.read_csv(restaurant_file)
        self.classifier = pkl.load(open(classifier_file,'rb'))
        self.possible_food_type = list(self.restaurant_data['food'].unique())
        self.possible_area = list(self.all_restaurants["area"].unique())
        self.possible_pricerange = list(self.all_restaurants["pricerange"].unique())
    def parseText(self,sentence):
        cls = self.classifier.predict(sentence)
        print(cls)


class RestaurantAgent(StateMachine):
    #states
    hello = State(initial=True)
    waiting_for_area = State()
    return_count = State()
    completed = State(final=True)
    #transitions
    receive_area = (
        waiting_for_area.to(return_count, cond="area_known")
        | waiting_for_area.to(waiting_for_area, unless="area_known")
    )
    start_processing = hello.to(waiting_for_area)
    complete_process = return_count.to(completed)
    

    def __init__(self,filename):
        self.tmpArea = "" #area input variable is stored here
        self.area = "" #if area is valid, area is stored here
        self.all_restaurants = pd.read_csv(filename)
        self.possible_areas = list(self.all_restaurants["area"].unique())
        super(RestaurantAgent, self).__init__()
    
    def count_restaurants(self):
        return len(self.all_restaurants[self.all_restaurants["area"]==self.area])
    
    def area_known(self, area):
        print(f"Looking for restaurants in {area},{self.possible_areas}")
        return area in self.possible_areas
    
    def before_receive_area(self,area):
        print("received area", area)
        self.area = area

    def on_enter_hello(self):
        print("Welcome to the UU-Restaurant Recommendation System, how can we help you today?")
    
    def on_enter_waiting_for_area(self):
        print("Which area of town would you like to eat in?")
    
    def on_enter_return_count(self):
        count = self.count_restaurants()
        print(f"There are {count} restaurant in the {self.area}-area of town")

restaurant_file = "restaurant_info.csv"
classifier_file = ".ass_1a/models/complete/DecisionTree.pkl"

parser = TextParser(classifier_file,restaurant_file)
parser.parseText("I am looking for a cheap chineese restaurant in the south of town")

"""
sm = RestaurantAgent(restaurant_file)
print(sm.current_state)
sm.send("start_processing")
print(sm.current_state)
sm.send("receive_area",area="southeast")
print(sm.current_state)
"""


