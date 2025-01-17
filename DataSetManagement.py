from sklearn.model_selection import train_test_split
import pandas as pd

class CurrentDataSets:
    dataSet = {"first":"BD1.txt","nigerian":"DataSmsSpamNH.csv","telegram":"telegram_spam_dataset.csv"}

class BDSpam:
    def __init__(self,randomState,name,BDpath):
        self.dataSet = pd.read_csv(BDpath,sep="\t",header=None,names=["spam","text"])
        self.name =name
        self.dataSet[""]
        X = self.dataSet["text"]
        y = self.dataSet["spam"].apply(str.lower)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y , test_size=0.2, random_state=randomState)

    def GetAll(self):
        return self.X_train,self.X_test,self.y_train,self.y_test

    def GetTest(self):
        return self.X_test,self.y_test
    
    def GetTrain(self):
        return self.X_train,self.y_train
    
class DataSetManage():
    def __init__(self,dataSet,randomState):
        self.BD = {}
        self.dataSetSelect = dataSet
        self.randomState = randomState
        for name,path in self.dataSetSelect.item:
            self.BD[name]= BDSpam(randomState=self.randomState,name=name,BDpath=path)

    def GetTrain(self,name):
        if name in self.BD.keys():
            return self.BD[name]
    
    def GetCombinedTrain(self,keys):
        TrainConcat = []
        TargetConcat = []
        for key in keys:
            if key not in self.BD:
                print("The key is not in the dataSet available")
            bd = self.BD[key]
            xtrain , ytrain = bd.GetTrain()
            TrainConcat.append(xtrain)
            TargetConcat.append(ytrain)
        TrainConcat = pd.concat(TrainConcat,ignore_index=True)
        TargetConcat = pd.concat(TargetConcat,ignore_index=True)
        return TrainConcat,TargetConcat

    def GetTestDF(self):
        retour = []
        for bd in self.BD.values():
            retour.append(bd.GetTest())
        return retour