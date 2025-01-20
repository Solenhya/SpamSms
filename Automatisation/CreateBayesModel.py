import pandas as pd
import joblib
import argparse


# Create the parser
parser = argparse.ArgumentParser(prog="BayesienCreation",description="Creer un modele bayesien avec une base de donnée sous format csv séparer par tabulation en premier l'étiquette 'ham' ou 'spam' et le contenue du sms")

# Add arguments
parser.add_argument('--TFile', type=str, help="The path to the training data", required=True)
parser.add_argument('--OFile',type = str,help="The name of the outputfile")
# Parse arguments
args = parser.parse_args()

#Lis le fichier de training data
trainDF = pd.read_csv(
    f"{args.TFile}", delimiter="\t", header=None, names=["spam", "text"]
)

#Handle les cas ou l'étiquette n'a pas la meme casse
trainDF["spam"]=trainDF["spam"].apply(str.lower)

#Creer un tableau des mots et de leur frequence
trainDFSpam = trainDF[trainDF["spam"]=="spam"]
trainDFSpam["text"]=trainDFSpam["text"].str.lower()
dfSplitSpam = trainDFSpam.text.str.split(expand=True).stack().value_counts().to_frame()

trainDFHam = trainDF[trainDF["spam"]=="ham"]
trainDFHam["text"]=trainDFHam["text"].str.lower()
dfSplitHam = trainDFHam.text.str.split(expand=True).stack().value_counts().to_frame()

#Les join et calcul la vraisamble ainsi que la probabilité de base d'un spam
modelFrame = dfSplitSpam.join(dfSplitHam,lsuffix='_spam', rsuffix='_ham')
modelFrame.fillna(0.001,inplace=True)
spamTotal=modelFrame["count_spam"].sum()
hamTotal = modelFrame["count_ham"].sum()
modelFrame["vraisemblance"]=(modelFrame["count_spam"]/spamTotal)/(modelFrame["count_ham"]/hamTotal)
spamProbality = spamTotal/hamTotal

##Change posible to reduce what we serialise to only what is needed

#On serialise le model
filepath=""
if args.OFile:
    filepath =args.OFile+'.joblib'   
else:
    filepath="./Models/BayesienModel.joblib"
joblib.dump((modelFrame,spamProbality),filepath)