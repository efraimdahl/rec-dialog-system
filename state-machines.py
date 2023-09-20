



#Basic State machine that returns the number of restaurants in an area of town given the keyword south, west, east, north and center.

from statemachine import StateMachine, State
import pandas as pd
import pickle as pkl

#The textParser receives a trained classifier as input, and will return packets such as 
# 
#inform(type=restaurant,pricerange=moderate,task=find)
class TextParser():
    def __init__(self,classifier_file, restaurant_file,vectorizer_file):
        self.restaurant_data = pd.read_csv(restaurant_file)
        self.classifier = pkl.load(open(classifier_file,'rb'))
        self.vectorizer = pkl.load(open(vectorizer_file,'rb'))
        self.possible_food_type = list(self.restaurant_data['food'].unique())
        self.possible_area = list(self.restaurant_data["area"].unique())
        self.possible_pricerange = list(self.restaurant_data["pricerange"].unique())
        self.dontCarePhrases={
            "area":["anywhere", "any where", "any part", "any area", "other area", "other part", "rest of town","any address","any locat"],
            "price":["any range","any price","dont care about the price","dont care about price","not care about the price","not care about price","dont care what price","price does not matter"],
            "food":["any kind","any food","any type", "any foo","another type"],
            "generic":["any","dont care","doesnt matter","dont know", "do not care","does not matter","doesnt matter","doesnt mater","deosnt matter","either one","surprise me","not important","dont mind", "what ever","dont matter"]
        }
        self.matchingRequestDict = {
            "phone":["number","phone"],
            "postcode":["postcode","post code", "post number","postal code"],
            "addr":["address"],
            "food" :["food","serve","what type of venue"],
            "pricerange" :["price","cost","how much"],
            "restaurantname":["name","called"],
            "area":["area","part","locat"]
        }
    def keywordMatcher(self,sentence):
        foodType = ""
        priceRange = ""
        area = ""
        words = sentence.split(" ")
        for word in words:
            #TODO Levenstein distance here
            if(word in self.possible_food_type):
                foodType = word
            if(word in self.possible_area):
                area = word
            if(word in self.possible_pricerange):
                priceRange = word
        return foodType,priceRange,area
    
    def anyDetector(self,sentence,context):
        phrases = self.dontCarePhrases.get(context)
        found = False
        for keyPhrase in phrases:
            if(keyPhrase in sentence):
                found = True
                break
        return found
    def requestMatching(self,sentence):
        ret = []
        for column in self.restaurant_data.columns:
            matcher = self.matchingRequestDict.get(column)
            for keyPhrase in matcher:
                if(keyPhrase in sentence):
                    ret.append(column)
                    break
        return ret

    def parseText(self,sentence,context=None,requestPossible=True): #the context variable is used to understand short answers such as any to the question "Where do you want to eat"
        #First get the classification of the utterance
        vec = self.vectorizer.transform([sentence])
        cls = self.classifier.predict(vec)
        #information classified as request, inform, confirm, negate, and reqalts can be overloaded with information from the sentence
        foodType,priceRange,area="","",""
        if(cls[0] in ["inform","reqalts","confirm","negate"] or (cls[0]=="request" and not requestPossible)):
            if(context==None):
                foodType,priceRange,area = self.keywordMatcher(sentence)
                if(area=="" and self.anyDetector(sentence,"area")):
                    area = "dontcare"
                if(foodType=="" and self.anyDetector(sentence,"food")):
                    foodType = "dontcare"
                if(priceRange=="" and self.anyDetector(sentence,"price")):
                    priceRange = "dontcare"
            else:
                doesntCare = self.anyDetector(sentence,"generic")
                if(doesntCare):
                    if(context=="area"):
                        area="dontcare"
                    if(context=="food"):
                        foodType="dontcare"
                    if(context=="price"):
                        priceRange=="dontcare"
            retlist = []
            names = ["foodType","priceRange","area"]
            vars = [foodType,priceRange,area]
            for i in range(0,len(vars)):
                if i != "":
                    retlist.append((names[i],vars[i]))
            return (cls[0],retlist)
        elif(cls[0] in ["request"]):
            requVars = self.requestMatching(sentence)
            return (cls[0],requVars)

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
classifier_file = "./ass_1a/models/complete/Ridge.pkl"
vectorizer_file = "./ass_1a/models/complete/vectorizer.pkl"
parser = TextParser(classifier_file,restaurant_file,vectorizer_file)
print(parser.parseText("I am looking for a moderate priced thai place located in the city centre"))
print(parser.parseText("Whats the price?"))
print(parser.parseText("I am looking for a moderate priced thai place located in any part of town"))
print(parser.parseText("I am looking for a place that serves any type of food in any part of town in any price range",requestPossible=False))

"""
sm = RestaurantAgent(restaurant_file)
print(sm.current_state)
sm.send("start_processing")
print(sm.current_state)
sm.send("receive_area",area="southeast")
print(sm.current_state)
"""


