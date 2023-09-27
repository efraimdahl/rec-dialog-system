import pandas as pd
import pickle as pkl
import math
import random
import Levenshtein

from typing import Union, Tuple
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import ClassifierMixin

from ass_1a.keyword_model import KeywordClassifier


class TextParser():
    """
    The textParser receives a trained classifier as input, and performs keyword matching to extract information of a sentence based on its classification.

    example: "I am looking for a cheap thai place in the south of town" -> inform=>[(pricerange=cheap),(food=thai),(area=south)]
    """
    def __init__(self, classifier: Union[ClassifierMixin, KeywordClassifier], restaurant_file: str, vectorizer: CountVectorizer) -> None:
        """
        Initializes a TextParser instance.
        Args:
            classifier (Union[ClassifierMixin, KeywordClassifier]): The classifier to be used by the parser
            restaurant_file (str): A string pointing to the restaurant file
            vectorizer (CountVectorizer): The trained CountVectorizer instance to be used.
        """
        self.restaurant_data = pd.read_csv(restaurant_file)
        self.classifier = classifier
        self.vectorizer = vectorizer 
        self.possible_food_type = list(self.restaurant_data['food'].unique())
        self.possible_area = list(self.restaurant_data["area"].unique())
        self.possible_pricerange = list(self.restaurant_data["pricerange"].unique())
        self.used_levenshtein = False
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
        
        
    def keywordMatcher(self, sentence: str) -> Tuple[str, str, str]:
        """ Matches keywords in a sentence to the possible outcomes for area,types and priceranges

        Args:
            sentence (str): The sentence to be matched

        Returns:
            Tuple[str, str, str]: A tuple containing the matched area, type and pricerange
        """
        foodType = ""
        priceRange = ""
        area = ""
        words = sentence.split(" ")
        threshold = 3   #Used for calculate the Levenstein distance
        all_keywords = self.possible_food_type + self.possible_area + self.possible_pricerange
        all_keywords = [x for x in all_keywords if isinstance(x, str)] #Delete None value
        for word in words:
            similar_keywords = []
            for keyword in all_keywords:
                distance = Levenshtein.distance(word, keyword)
                if distance <= threshold:
                    similar_keywords.append(keyword)
                    self.used_levenshtein = True
                elif distance == 0:
                    similar_keywords = [word] #Fonund the identical word and break, override the keywords
                    self.used_levenshtein = False
                    break
            if len(similar_keywords) == 0:
                similar_keywords.append(word)
            word = random.choice(similar_keywords)
            if(word in self.possible_food_type):
                foodType = word
            if(word in self.possible_area):
                area = word
            if(word in self.possible_pricerange):
                priceRange = word
        return foodType, priceRange, area
    

    def anyDetector(self, sentence: str, context: str) -> bool:
        """Detects "i don't care" instances both in a generic setting, or in relation to a pricerange, area or food-type.

        Args:
            sentence (str): The sentence to be matched
            context (str): The context in which the sentence is used, can be "area", "priceRange", "foodType" or "generic"

        Returns:
            bool: True if the sentence contains "i don't care" or similar, False otherwise
        """
        phrases = self.dontCarePhrases.get(context)
        for keyPhrase in phrases:
            if(keyPhrase in sentence):
                return True
        return False
    
    
    def requestMatching(self, sentence: str) -> list:
        """For requests, where the user is not providing information, find the keywords that indicate what the user is looking for.

        Args:
            sentence (str): The sentence to be matched

        Returns:
            list: A list containing the matched requests
        """
        ret = []
        for column in self.restaurant_data.columns:
            matcher = self.matchingRequestDict.get(column)
            for keyPhrase in matcher:
                if(keyPhrase in sentence):
                    ret.append(column)
                    break
        return ret
    
    
    
    def parseText(self, sentence: str, context: Union[str, None]=None, requestPossible: bool=True) -> Tuple[str, dict]: 
        """Text Parsing function takes the sentence as input, classifies it, and retrieves information, if its classified as containing information.
        The context variable is passed  to understand short answers such as "any" to the system-prompt "Where do you want to eat"
        Request possible is passed because in practice many inform instances are classified as requests, which are only possible after a restaurant is suggested.
        we are trying to circumvent this missclassification to improve usability.

        Args:
            sentence (str): The sentence to be parsed
            context (Union[str, None], optional): The context in which the sentence is used, can be "area", "priceRange", "foodType" or "generic". Defaults to None.
            requestPossible (bool, optional): Whether requests are possible. Defaults to True.

        Returns:
            Tuple[str, dict]: A tuple containing the classification and the information extracted from the sentence
        """
        # First get the classification of the utterance
        sentence = sentence.lower()
        vec = self.vectorizer.transform([sentence])
        cls = self.classifier.predict(vec)

        # Information classified as inform, confirm, negate, and reqalts can be overloaded with information from the user-sentence
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
            return (cls[0],retlist), self.used_levenshtein
        #What fields are users requesting?
        elif(cls[0] in ["request"]):
            requVars = self.requestMatching(sentence)
            return ([cls[0],requVars]), self.used_levenshtein
        else:
            return([cls[0]]), self.used_levenshtein

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