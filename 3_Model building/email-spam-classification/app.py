import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


def preprocess_text(text):
    # lower text
    text = text.lower()
    
    # tokenization
    text = nltk.word_tokenize(text)
    
    # remove special characters
    text = [i for i in text if i.isalnum()]
    
    # remove stop words & punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    
    # stemming
    ps = PorterStemmer()
    text = [ps.stem(i) for i in text]
    
    return " ".join(text)

model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

st.title('SMS/Email Spam Classifier')

email = st.text_area('Enter the message')

if st.button('Predict'):
    print(email)
    # preprocess
    transformed_text = preprocess_text(email)
    print(transformed_text)
    # vectorize
    vector_input = tfidf.transform([transformed_text])
    print(vector_input)
    # predict
    prediction = model.predict(vector_input)[0]
    # display
    if prediction == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')

