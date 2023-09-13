from collections import Counter
import pandas as pd

class FrequencyModel():
    """
    Predicts the most frequent label in the data
    """
    def __init__(self):
        pass
    
    def fit(self,X,y):
        self.pred = y.value_counts().idxmax()
        
    def predict(self,_):
        return self.pred

class 