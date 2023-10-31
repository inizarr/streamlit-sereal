import pickle
import streamlit as st

# Muat model
model = pickle.load(open('Sereal.sav', 'rb'))

st.title('Prediksi Sereal')

calories = st.number_input('Input calories')
protein = st.number_input('Input protein')
fat = st.number_input('Input fat')
sodium = st.number_input('Input sodium')
fiber = st.number_input('Input fiber')
carbo = st.number_input('Input carbo')
sugars = st.number_input('Input sugars')
rating = st.number_input('Input rating')

predict = ''

if st.button('Prediksi Sereal'):
    input_data = [[calories, protein, fat, sodium, fiber, carbo, sugars, rating]]
    predict = model.predict(input_data)
    st.write('Prediksi Sereal:', predict)
