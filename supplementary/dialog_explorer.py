
f = open("data/all_dialogs.txt", "r")
l = f.readlines()

#Look for any case of data where the data is missing. 

def keyPhraseFinder(l,variable,keyPhrases):
    count=0
    for i in range(0,len(l)):
        sentence=l[i]
        if(variable in sentence):
            keyphraseFound = 0
            for k in keyPhrases:
                if(k in l[i-1]):
                    keyphraseFound=1
            if(keyphraseFound == 0):
                print(sentence)
                print(l[i-1])
                count +=1
    print("Instance:", count)

keyAreaPhrases = ["anywhere", "any where", "any part", "any area", "other area", "other part", "rest of town","any address"]
keyFoodPhrases = ["any kind","any food","any type", "any foo","another type"]
keyPricePhrases = ["any range","any price","dont care about the price","dont care about price","not care about the price","not care about price","dont care what price","price does not matter"]
dontCarePhrases = ["any","dont care","doesnt matter","dont know", "do not care","does not matter","doesnt matter","doesnt mater","deosnt matter","either one","surprise me","not important","dont mind", "what ever","dont matter"]

requestPhoneKeywords = ["number","phone"]
requestPostCode = ["postcode","post code", "post number","postal code"]
requestAddressKeys = ["address"]
requestFood = ["food","serve","what type of venue"]
requestPrice = ["price","cost","how much"]
requestName = ["name","called"]
requestArea = ["area","part","locat"]
#keyPhraseFinder(l,"area=dontcare",keyAreaPhrases)
#keyPhraseFinder(l,"food=dontcare",keyFoodPhrases)
#keyPhraseFinder(l,"pricerange=dontcare",keyPricePhrases)
#keyPhraseFinder(l,"(=dontcare)",dontCarePhrases)
keyPhraseFinder(l,"request(area",requestArea)