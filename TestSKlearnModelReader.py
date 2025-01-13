from SKLearnModels import NaiveBayesModel
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("BD1.txt", sep="\t", names=["spam", "text"])

model = NaiveBayesModel(df, random=None)
print(classification_report(model.y_test, model.Predict()))

print(model.Predict(["You won a free holiday to the Maldives! Call 09066364304 now!!"]))
