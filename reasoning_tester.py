import json
import pandas as pd

reasoning_file = "data/reasoning_rules.json"
restaurant_file = "data/restaurant_info.csv"

reasoning_rules = json.loads(open(reasoning_file,"rb").read())


def search_restaurant(area,foodType,priceRange) -> pd.DataFrame:
        """Searches the restaurant database for restaurants matching the current variables

        Returns:
            pd.DataFrame: A dataframe containing the restaurants matching the current variables
        """
        df = pd.read_csv(restaurant_file)
        if(area!="dontcare"):
            df = df[df["area"]==area]
        if(priceRange!="dontcare"):
            df=df[df["pricerange"]==priceRange]
        if(foodType!="dontcare"):
            df=df[df["food"]==foodType]
        return df

def reasoner(qualifier,area,foodType,priceRange):

    neg_cond = reasoning_rules[qualifier+"-false"]
    pos_cond = reasoning_rules[qualifier+"-true"]

    print(pos_cond,neg_cond)
    use_description = ""
    df = search_restaurant(area,foodType,priceRange)
    pdf = df #Dataframe with positive adding qualifiers
    ndf = df #Dataframe with non-negative qualifiers
    for cond in pos_cond["conditions"]:
        print(cond)
        pdf=df[df[cond[0]]==cond[1]]
    for cond in neg_cond["conditions"]:
        print(cond)
        ndf=df[df[cond[0]]!=cond[1]]
    #Restaurants with only positive qualifiers receive preferencial treatment
    #print(pdf)
    res_df = pd.merge(pdf, ndf, how ='inner')
    #print(len(pdf),len(ndf),len(res_df))
    use_description = pos_cond["description"]
    if(len(res_df)==0):
        #restaurants with non-negative attributes come second
        res_df = ndf
        use_description=neg_cond["description"]
        if(len(ndf)==0):
            #restaurants with positive attributes and negative attributes come third - no need to complete an outer merge here
            res_df = pdf
            use_description = pos_cond["description"]
    if(len(res_df)>0):
        print(res_df,use_description)

reasoner("romantic","south","italian","dontcare")
reasoner("romantic","dontcare","dontcare","dontcare")