from re import X
from SKLearnModels import NaiveBayesModel, SCVModel, RandomForestModel, LogisticRegressionModel, NaiveBayesFeaturesModel
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv("BD1.txt", sep="\t", names=["spam", "text"])
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["spam"], test_size=0.2, random_state=42)

print("---------------------------------------------")
model = NaiveBayesModel(X_train,y_train, random=42)
print("NaiveBayesModel")
print(classification_report(y_test, model.Predict(X_test)))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("---------------------------------------------")
model = NaiveBayesFeaturesModel(X_train,y_train, random=42)
print("NaiveBayesFeaturesModel")
print(classification_report(y_test, model.Predict(X_test)))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("----------------------------------------------")
print("LinearSVCModel")
model = SCVModel(X_train,y_train, random=42)
print(classification_report(y_test, model.Predict(X_test)))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("----------------------------------------------")
print("RandomForestModel")
model = RandomForestModel(X_train,y_train, random=42)
print(classification_report(y_test, model.Predict(X_test)))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))

print("----------------------------------------------")
print("LogisticRegressionModel")
model = LogisticRegressionModel(X_train,y_train, random=42)
print(classification_report(y_test, model.Predict(X_test)))
print(model.Predict(test_data=["You won a free holiday to the Maldives! Call 09066364304 now!!"]))
