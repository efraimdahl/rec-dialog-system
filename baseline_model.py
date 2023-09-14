from collections import Counter
import pandas as pd
import numpy as np

class KeywordClassifier():
    """
    Predicts the label using a rule-based system based on keyword matching
    """
    def __init__(self):
        # There is no learning, we just initialize the keyword matching dictionary
        # I have omitted most rules for assigning the "inform" label because it will be the base if nothing else matches
        # The order of the rules matters because the first one is chosen.
        self.keywords = {
            "thank" : "thankyou",
            "goodbye" : "bye",
            "good bye" : "bye",
            "looking for" : "inform",
            "could" : "request",
            "address" : "request",
            "post code" : "request",
            "postcode" : "request",
            "phone" : "request",
            "can" : "request",
            "else" : "reqalts",
            "type" : "request",
            "yes" : "affirm",
            "yea" : "affirm",
            "how" : "reqalts",
            "kay" : "ack",
            "is it" : "confirm",
            "does it" : "confirm",
            "wrong" : "deny",
            #"dont" : "deny",
            #"no not" : "deny",
            "hello" : "hello",
            "hi" : "hello",
            "noise" : "null",
            "no" : "negate",
            #"cough" : "null",
            "repeat" : "repeat",
            "what about" : "reqalts",
            "more" : "reqmore",
            "start" : "restart",
            "" : "inform" # Catchall for inform
        }
    
    def fit(self,X,y):
        pass
        
    def predict(self,X):
        # Note that X should not be vectorised
        labels = []
        for text in X:
            for key in self.keywords.keys():
                if key in text:
                    print(key)
                    print(self.keywords[key])
                    labels.append(self.keywords[key])
                    break
        return labels
    
    def score(self,X,y):
        # Simple accuracy score
        return sum(self.predict(X) == y)/len(y)
        
        