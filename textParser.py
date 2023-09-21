import pandas as pd
import pickle as pkl

#The textParser receives a trained classifier as input, and perform keyword matching to extract information of a sentence based on its classification.
# 
#example: "I am looking for a cheap thai place in the south of town" -> inform=>[(pricerange=cheap),(food=thai),(area=south)]
class TextParser():
    def __init__(self,classifier,restaurant_file,vectorizer):
        self.restaurant_data = pd.read_csv(restaurant_file)
        self.classifier = classifier
        self.vectorizer = vectorizer 
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
    #Keyword-matching based on possible outcomes for area,types and priceranges
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
    #Detects "i don't care" instances both in a generic setting, or in relation to a pricerange,area or food-type.
    def anyDetector(self,sentence,context):
        phrases = self.dontCarePhrases.get(context)
        found = False
        for keyPhrase in phrases:
            if(keyPhrase in sentence):
                found = True
                break
        return found
    #For requests, where the user is not providing information, find the keywords that indicate what the user is looking for.
    def requestMatching(self,sentence):
        ret = []
        for column in self.restaurant_data.columns:
            matcher = self.matchingRequestDict.get(column)
            for keyPhrase in matcher:
                if(keyPhrase in sentence):
                    ret.append(column)
                    break
        return ret
    # Text Parsing function takes the sentence as input, classifies it, and retrieves information, if its classified as containing information.
    #The context variable is passed  to understand short answers such as "any" to the system-prompt "Where do you want to eat"
    #Request possible is passed because in practice many inform instances are classified as requests, which are only possible after a restaurant is suggested.
    #we are trying to circumvent this missclassification to improve usability.
    def parseText(self,sentence,context=None,requestPossible=True): 
        #First get the classification of the utterance
        sentence = sentence.lower()
        vec = self.vectorizer.transform([sentence])
        cls = self.classifier.predict(vec)
        print(cls[0])
        #information classified as inform, confirm, negate, and reqalts can be overloaded with information from the user-sentence
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
                foodType,priceRange,area = self.keywordMatcher(sentence)
                doesntCare = self.anyDetector(sentence,"generic")
                if(doesntCare):
                    if(context=="area"):
                        area="dontcare"
                    if(context=="foodType"):
                        foodType="dontcare"
                    if(context=="priceRange"):
                        priceRange="dontcare"

            retlist = {}
            names = ["foodType","priceRange","area"]
            vars = [foodType,priceRange,area]
            for i in range(0,len(vars)):
                if vars[i] != "":
                    retlist.update({names[i]:vars[i]})
            return (cls[0],retlist)
        #What fields are users requesting?
        elif(cls[0] in ["request"]):
            requVars = self.requestMatching(sentence)
            return ([cls[0],requVars])
        else:
            return([cls[0]])

def main():
    restaurant_file = "restaurant_info.csv"
    classifier = pkl.load(open("./ass_1a/models/complete/DecisionTree.pkl",'rb'))
    vectorizer = pkl.load(open("./ass_1a/models/complete/vectorizer.pkl",'rb'))
    parser = TextParser(classifier,restaurant_file,vectorizer)
    print(parser.parseText("Hello"))
    print(parser.parseText("how about a cheap asian oriental restaurant"))
    print(parser.parseText("how about french food"))
    print(parser.parseText("any more"))
    print(parser.parseText("I am looking for a moderate priced thai place located in the city centre"))
    print(parser.parseText("Whats the price?"))
    print(parser.parseText("I am looking for a moderate priced thai place located in any part of town"))
    print(parser.parseText("I am looking for a place that serves any type of food in any part of town in any price range",requestPossible=False))
    print(parser.parseText("I dont care",context="area",requestPossible=False))
    print(parser.parseText("I dont care",context="foodType",requestPossible=False))
    print(parser.parseText("I dont care",context="priceRange",requestPossible=False))



if __name__ == '__main__':
    main()