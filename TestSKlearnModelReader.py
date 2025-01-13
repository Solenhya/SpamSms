from SKLearnModels import NaiveBayesModel, SCVModel, RandomForestModel, LogisticRegressionModel, NaiveBayesFeaturesModel
import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv("BD1.txt", sep="\t", names=["spam", "text"])

print("---------------------------------------------")
model = NaiveBayesModel(df, random=42)
print("NaiveBayesModel")
print(classification_report(model.y_test, model.Predict()))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("---------------------------------------------")
model = NaiveBayesFeaturesModel(df, random=42)
print("NaiveBayesFeaturesModel")
print(classification_report(model.y_test, model.Predict()))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("----------------------------------------------")
print("LinearSVCModel")
model = SCVModel(df, random=42)
print(classification_report(model.y_test, model.Predict()))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("----------------------------------------------")
print("RandomForestModel")
model = RandomForestModel(df, random=42)
print(classification_report(model.y_test, model.Predict()))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("----------------------------------------------")
print("LogisticRegressionModel")
model = LogisticRegressionModel(df, random=42)
print(classification_report(model.y_test, model.Predict()))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))
