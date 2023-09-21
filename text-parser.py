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
        else:
            return(cls[0])

restaurant_file = "restaurant_info.csv"
classifier_file = "./ass_1a/models/complete/Ridge.pkl"
vectorizer_file = "./ass_1a/models/complete/vectorizer.pkl"
parser = TextParser(classifier_file,restaurant_file,vectorizer_file)
print(parser.parseText("I am looking for a moderate priced thai place located in the city centre"))
print(parser.parseText("Whats the price?"))
print(parser.parseText("I am looking for a moderate priced thai place located in any part of town"))
print(parser.parseText("I am looking for a place that serves any type of food in any part of town in any price range",requestPossible=False))
print(parser.parseText("I dont care",context="area",requestPossible=False))