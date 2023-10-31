import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats

filename = "cleanedQuestionnaire.csv"
df = pd.read_csv(filename)
question_map = pd.read_csv("Question Mapping.csv",header=None)
question_map = dict(zip(question_map[0],question_map[1]))

qA = [question + "-A" for question in question_map.keys()]
qB = [question + "-B" for question in question_map.keys()]

for A,B in zip(qA,qB):
    sns.violinplot(data=df[[A,B]])
    sns.boxplot(data=df[[A,B]], saturation=0.3, width=0.2, boxprops={'zorder': 2})
    plt.ylim((1,5))
    plt.ylabel("Likert score")
    plt.xlabel("System")
    plt.title(question_map[A[:-2]])
    plt.show()
    
t_results = {}
for N,A,B in zip(question_map.keys(),qA,qB): 
    t_test = stats.ttest_rel(df[A],df[B])[1]
    t_results[N] = t_test
    print(f"p_value for the question \"{question_map[N]}\": {t_test}")
    
    


