
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the model and other necessary files
model = pickle.load(open("./passmodel.pkl","rb"))    #check your directory
vectorizer = pickle.load(open("./tfidfvectorizer.pkl","rb"))    #check your directory
df=pd.read_csv("./dataset.csv")


stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Set the page configuration with the title and background image
st.set_page_config(
    page_title="Condition and Drug Name Prediction",
    page_icon=":pill:",
    layout="wide",
    initial_sidebar_state="expanded"
)

import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background('./background.jpg')




html_temp="""
<div style ="background-color:Black;padding:10px">
<h2 style="color:white;text-align:center;"> Condition and Drug Name Prediction </h2>
"""
st.markdown(html_temp,unsafe_allow_html=True)


# Define function to clean and preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if not word in stop_words]
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

def top_drugs_extractor(condition):
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==condition]['drugName'].head(3).tolist()
    return drug_lst
def predict_text(lst_text):
    df_test = pd.DataFrame(lst_text, columns = ['test_sent'])
    df_test["test_sent"] = df_test["test_sent"].apply(review_to_words)
    tfidf_bigram = tfidf_vectorizer3.transform(lst_text)
    prediction = pass_tf.predict(tfidf_bigram)
    df_test['prediction']=prediction
    return df_test
# Create text input for user to enter review
text = st.text_input('Enter the Text: ')

if st.button('Predict'):
    test = vectorizer.transform([text])
    pred1 = model.predict(test)[0]
    st.subheader("Condition:")
    st.write(pred1)
    
    df_top = df[(df['rating']>=9)&(df['usefulCount']>=100)].sort_values(by = ['rating', 'usefulCount'], ascending = [False, False])
    drug_lst = df_top[df_top['condition']==pred1]['drugName'].head(3).tolist()
    st.subheader("Recommended Drugs:")
    for i, drug in enumerate(drug_lst):
        st.write(i+1, drug)
    
