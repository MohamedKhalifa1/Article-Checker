
import joblib

import streamlit as st

import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
st.header('Text Checker')

nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

model = joblib.load("model.pkl")
vector_form= joblib.load("vector.pkl")


stemmer=PorterStemmer()


def word_cleaning(article):
    article=re.sub('[^a-zA-Z]',' ',article)
    article=article.lower()
    article=article.split()
    article=[stemmer.stem(sentences)for sentences in article if not sentences in stopwords.words('english')]
    article=" ".join(article)
    return article


def response(article):
    article = word_cleaning(article)
    input_data = [article]
    vector_form1= vector_form.transform(input_data)
    prediction = model.predict(vector_form1)
    return prediction


if __name__ == '__main__':
    st.title('Check AI Text Generated app ')
    st.subheader("Input the Article content below")
    sentence = st.text_area("Enter your Article content here", "",height=200)
    predict_btt = st.button("predict")
    if predict_btt:
        prediction_class=response(sentence)
        print(prediction_class)
        if prediction_class == [0]:
            st.success('Human Writer')
        if prediction_class == [1]:
            st.warning('Ai Generated')