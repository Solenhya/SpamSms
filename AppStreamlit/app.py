import streamlit as st

def detect_spam(sms):
    if sms == "":
        return False
    else:
        # Todo: Implement spam detection logic
        return True
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
        is_spam = detect_spam(sms)
        
if is_spam:
    st.error("The sms is a spam")
else:
    st.success("The sms is not a spam")

