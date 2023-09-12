import sklearn
import numpy as np
import pandas as pd
import baseline_model
labels = []
messages = []
with open("dialog_acts.dat") as f:
    for line in f.readlines():
        label = line.split(" ")[0]
        labels.append(label)
        messages.append(line[len(label)+1:-1])
df = pd.DataFrame(list(zip(labels,messages)),columns =['Label',"Text"])
#Act_Dict : {"ack" : 0, "affirm" : 1, "bye" : 2, "confirm" : 3 , "deny" : 4, "hello" : 5, "inform" : 6, "negate" : 7, "null" : 8, "repeat" : 9, "null" : 10, "repeat": 11, "reqalts" : 12, "reqmore" : 13, "request" : 14, "restart" : 15, "thankyou" : 16}
train, test = sklearn.model_selection.train_test_split(df, test_size=0.2)
model1 = baseline_model.FrequencyModel()
model1.fit(df['Text'],df['Label'])