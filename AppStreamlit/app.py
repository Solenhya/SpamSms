import streamlit as st
import pandas as pd
import joblib

loaded_df, spamProbality = joblib.load('./BayesienModel.joblib')

def TestPhraseList(phraseList,spamProba=1):
    for word in phraseList:
        vraisemblance = loaded_df["vraisemblance"].get(word,"NotFound")
        if vraisemblance!="NotFound":
            spamProba*=vraisemblance
    return(spamProba)

def detect_spam(sms):
    if sms == "":
        return False
    else:
        spamProbality = TestPhraseList(sms.split())
        return spamProbality
is_spam = False

st.set_page_config(
    page_title="Spam Detector",
    page_icon="🔕",
    menu_items={
    }
)

st.title("Spam Detector")
st.header("To detect if the sms is a spam")
sms = st.text_area("Enter the sms here", key="sms", height=150)
with st.columns(5)[2]:
    if st.button("Detect", use_container_width=True):
        spamProbality = detect_spam((sms))
        is_spam = spamProbality > 1
        
if is_spam:
    st.error(f"The sms is a spam, probability : {spamProbality}")
elif sms:
    st.success(f"The sms is not a spam, probability : {spamProbality}")

