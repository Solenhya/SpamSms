import streamlit as st
import pandas as pd
import joblib
import sys
import os
import re
from sklearn.svm import LinearSVC
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Automatisation import DataSetManagement
from Automatisation import EvaluateModelsFeatures

st.set_page_config(
    page_title="Spam Detector",
    page_icon="🔕",
    menu_items={
    }
)

@st.cache_resource
def TrainModel():
    dataSet = {
        "first":"../DataSetBrut/BD1.txt",
        "nigerian":"../DataSetBrut/DataSmsSpamNH.csv",
        "telegram":"../DataSetBrut/telegram_spam_dataset.csv"
    }
    dataManag = DataSetManagement.DataSetManage(dataSet,42)
    Xtrain,ytrain = dataManag.GetCombinedTrain(["first","nigerian","telegram"])
    features_dict = {
        "presence_monnaie": lambda x: 1 if re.search(r'[\$\€\£]', x) else 0 ,
        "presence_caratere_speciaux": lambda x: 1 if re.search(r'[!@#$%^&*(),.?":{}|<>]', x) else 0,
        "presence_lien": lambda x: 1 if re.search(r'\b(http|www)\S+', x) else 0
    }
    modelPipe = EvaluateModelsFeatures.GenerateModel(
        features_dict,
        model=LinearSVC(),
        X_train=Xtrain,
        y_train=ytrain
    )
    return modelPipe

modelPipe = TrainModel()

def detect_spam(sms):
    if sms == "":
        return False
    else:
        df = pd.DataFrame([sms], columns=["text"])
        return modelPipe.predict(X=df)[0]
is_spam = False


st.title("Spam Detector")
st.header("To detect if the sms is a spam")
sms = st.text_area("Enter the sms here", key="sms", height=150)
with st.columns(5)[2]:
    if st.button("Detect", use_container_width=True):
        spamResult = detect_spam((sms))
        is_spam = spamResult == "spam"
        
if is_spam:
    st.error(f"The sms is a spam")
elif sms:
    st.success(f"The sms is not a spam")

