from collections import Counter
import pandas as pd
import numpy as np

class KeywordClassifier():
    """
    Predicts the label of a sentence using a rule-based system based on keyword matching
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
            "hello" : "hello",
            "hi" : "hello",
            "noise" : "null",
            "no" : "negate",
            "repeat" : "repeat",
            "what about" : "reqalts",
            "more" : "reqmore",
            "start" : "restart",
            "" : "inform" # Catchall for inform
        }
    
    def fit(self,X,y):
        """
        Does nothing, just here for completeness.
        """
        pass
        
    def predict(self,X):
        """
        Predict the labels for a list of sentences using keyword matching.

        Parameters
        ----------
        X : List of sentences as strings.

        Returns
        -------
        labels : list of predicted labels
        """
        # Note that X should not be vectorised
        labels = []
        for text in X:
            for key in self.keywords.keys():
                if key in text:
                    labels.append(self.keywords[key])
                    break
        return labels
    
    def score(self,X,y):
        """
        Compares the predicted labels to the true labels and returns the accuracy
        """
        # Simple accuracy score
        return sum(self.predict(X) == y)/len(y)
        
        