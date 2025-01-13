from SpamSms.SKLearnModels import NaiveBayesModel
import pandas as pd

df = pd.read_csv("BD1.txt", sep="\t", names=["spam", "text"])

model = NaiveBayesModel(df)
print(model.Predict(["I am good"]))

