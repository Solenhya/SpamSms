import pandas as pd
import joblib

class BayesienModel:
    #Initialise le model depuis le joblib serialiser
    def __init__(self,modelPath):
        self.vraisemblanceDF , self.dataBias = joblib.load(modelPath)
    #Les definitions de fonction de test
    def TestPhraseList(self,phraseList,spamProba=1):
        for word in phraseList:
            vraisemblance = self.vraisemblanceDF["vraisemblance"].get(word,"NotFound")
            if vraisemblance!="NotFound":
                spamProba*=vraisemblance
        return(spamProba)
    def TestPhrase(self,phrase:str):
        phraseList=phrase.split()
        return self.TestPhraseList(phraseList,spamProba=self.dataBias)
    
    def Predict(self,phrase):
        value = self.TestPhrase(phrase)
        if(value<1):
            return "Ham"
        else:
            return "Spam"
    def Booleise(spam:str):
        spam = spam.lower()
        if(spam=="spam"):
            return 1
        if(spam=="ham"):
            return 0
    def Evaluate(self,brutDataSet:pd.DataFrame):
        brutDataSet[brutDataSet.columns[0]].apply(str.lower)
        brutDataSet["Result"]=brutDataSet[brutDataSet.columns[1]].apply(self.Predict)
        brutDataSet["Result"]=brutDataSet["Result"].apply(str.lower)
        return brutDataSet

