import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

class SpamAnalysis:
    def __init__(self,pathtoLoad):
        self.dataSet = pd.read_csv(pathtoLoad,sep="\t",header=None,names=["spam","text"])
        self.dataSet["spam"] = self.dataSet["spam"].apply(str.lower)
        self.dataSet["wordCount"]=self.WordCountCollumns(self.dataSet)
        self.dataSet["characCount"]=self.CharacCountCollumns(self.dataSet)
        self.dataSet["moneyChar"]=self.MoneyCharacCollumns(self.dataSet)
        self.dataSet["digitCount"]=self.NumbersCollumns(self.dataSet)
        self.dataSet["link"]=self.LinksCollumns(self.dataSet)

    def WordCountCollumns(self,dataSet:pd.DataFrame):
        retour=dataSet["text"].str.split()
        retour = retour.apply(len)
        return retour
    def CharacCountCollumns(self,dataSet:pd.DataFrame):
        retour=dataSet["text"]
        retour = retour.apply(len)
        return retour
    def MoneyCharacCollumns(self,dataSet:pd.DataFrame):
        retour = dataSet["text"]
        retour = retour.apply(lambda x :len(re.findall(r'[\$\€\£]',x)))
        return retour
    def NumbersCollumns(self,dataSet:pd.DataFrame):
        retour = dataSet["text"]
        retour = retour.apply(lambda x:len(re.findall(r'\d',x)))
        return retour
    def LinksCollumns(self,dataSet:pd.DataFrame):
        retour = dataSet["text"]
        retour = retour.apply(lambda x:len(re.findall(r'\b(http|www)',x)))
        return retour

    def PieOnFeature(self,feature,title=""):
        data=self.dataSet[["spam",feature]]
        dataSpam = data[data["spam"]=="spam"].groupby(feature).count()
        dataHam = data[data["spam"]=="ham"].groupby(feature).count()
        fig,axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].pie(dataSpam["spam"], labels=dataSpam.index,startangle=90,autopct=lambda x:f'{x:.1f}%')
        axes[0].set_title('ham')
        axes[1].pie(dataHam["spam"],labels=dataHam.index,startangle = 90,autopct=lambda x:f'{x:.1f}%')
        axes[1].set_title('spam')
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def BoiteMoustache(self,feature,title="",xlabel="",ylabel=""):
        data=self.dataSet[["spam",feature]]
        fig, ax = plt.subplots(figsize=(15, 5))
        sns.boxplot(x=feature, y="spam", data=data, orient='h', ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.show()


    def ShowClassification(self):
        dataShow = self.dataSet[["spam","text"]].groupby("spam").count()
        fig , ax = plt.subplots(figsize=(15, 5))
        ax.pie(dataShow["text"],labels=dataShow.index,startangle=90,autopct=lambda x:f'{x:.1f}%')
        plt.title("Repartition des données")
        plt.show()

    def GraphAnalysis(self):
        print(f"Nombre de données : {len(self.dataSet.index)}")
        self.ShowClassification()
        self.BoiteMoustache("wordCount",title="Nombre de mots par phrase",xlabel="Nombres",ylabel="Ham/Spam")
        self.BoiteMoustache("characCount",title="Nombre de characters par phrase",xlabel="Nombres",ylabel="Ham/Spam")
        self.PieOnFeature("moneyChar",title="Nombre de symbole monetaire dans ham ou spam")
        self.BoiteMoustache("digitCount",title="Nombre de chiffre par phrase",xlabel="Nombres",ylabel="Ham/Spam")
        self.PieOnFeature("link",title="Nombre de liens dans la phrase")