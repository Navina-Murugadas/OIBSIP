import streamlit as st
import pandas as pd
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

def main():
    st.title("SALES PREDICTOR")
    
    tv = st.number_input("Enter the TV value:")
    radio = st.number_input("Enter the Radio value:")
    newspaper = st.number_input("Enter the Newspaper value:")
    
    new_data = pd.DataFrame({
        "TV": [tv],
        "Radio": [radio],
        "Newspaper": [newspaper]
    })
    
    if st.button("PREDICT SALES"):
        new_predictions = model.predict(new_data[['TV', 'Radio', 'Newspaper']])
        st.subheader(f"Predicted Sales --> {new_predictions[0]:.2f}")

if __name__ == '__main__':
    main()
