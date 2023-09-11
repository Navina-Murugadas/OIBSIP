import streamlit as st
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

load_model = pickle.load(open('model.pkl','rb'))

st.title('IRIS FLOWER CLASSIFIER')

# Collect user input for prediction
sepal_length = st.number_input('Sepal Length (cm)')
sepal_width = st.number_input('Sepal Width (cm)')
petal_length = st.number_input('Petal Length (cm)')
petal_width = st.number_input('Petal Width (cm)')

if st.button('CLASSIFY'):
    user_input = [[sepal_length, sepal_width, petal_length, petal_width]]
    model = LogisticRegression()

    prediction = load_model.predict(user_input)

    iris_species = ["Setosa", "Versicolor", "Virginica"]
    predicted_species = iris_species[prediction[0]]
    
    st.subheader(f"PREDICTED IRIS SPECIES ---> {predicted_species}")