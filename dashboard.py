import streamlit as st
import pandas as pd

from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from lime import lime_text
import re

@st.cache_resource

def load_vect_and_model():
    text_vectorizer = load("C:/Users/Firoj Ansari/Downloads/Infosys_Dashboard/vetorizer.joblib")
    classif = load("C:/Users/Firoj Ansari/Downloads/Infosys_Dashboard/classif.joblib")

    return text_vectorizer, classif

text_vectorizer, classif = load_vect_and_model()

def vectorize_text(texts):
    text_transformed = text_vectorizer.transform(texts)
    return text_transformed

def pred_class(texts):
    return classif.predict(vectorize_text(texts))

def pred_probs(texts):
    return classif.predict_proba(vectorize_text(texts))

def create_colored_review(review, word_contributions):
    tokens = re.findall(text_vectorizer.token_pattern, review)
    modified_review = ""
    for token in tokens:
        if token in word_contributions["Word"].values:
            idx = word_contributions["Word"].values.tolist().index(token)
            conribution = word_contributions.iloc[idx]["Contribution"]
            modified_review += ":green[{}]".format(token) if conribution>0 else ":red[{}]".format(token)
            modifies_review += " "
        else:
            modified_review += token
            modified_review += " "

    return modified_review

exlainer = lime_text.LimeTextExplainer(class_names=classif.classes_)

st.image('infosys_img.png')
st.markdown("# Data Science Project")
st.markdown("### About the project :")
st.markdown("##### We have use deep learning models to predict if an given jop post is Fake or not. As there are a lot of fake job posting on the net which is there to only take advantage of the people who are unble to tell if an job post is real of fake. Also it is very difficult to idendify the fake job posts. So we have developed an model which can efficently and accurately identify a Fake job.")

st.title("Fake Job Posts Classification")

review = st.text_area(label="Enter Description Here:", value="text", height=20)
submit = st.button("Classify")

if submit and review:
    prediction, probs = pred_class([review,]), pred_probs([review,])
    prediction, probs = prediction[0], probs[0]

    st.markdown("### Prediction :")
    st.metric(label="Not Fake", value="{:.2f} %".format(probs[0]//10))
    st.metric(label="Fake", value="{:.2f} %".format(100-(probs[0]//10)))


