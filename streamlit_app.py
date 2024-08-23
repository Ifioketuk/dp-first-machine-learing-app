import streamlit as st
import requests
import pandas as pd
url = "https://raw.githubusercontent.com/Ifioketuk/dp-first-machine-learing-app/main/Nigerian Student's Year One Performance Survey(1-174).xlsx"
with open("Nigerian Student's Year One Performance Survey(1-174).xlsx.xlsx","wb") as file:
      file.write(response.content)
      
st.title('🤖 CGPA Prediction App')

st.write('This is an app used to predict the cgpa at the end of your first year based off academic performance at the end of high-school and study habits during the first year semesters.')
response= requests.get(url)


df= pd.read_excel(r"C:\Users\emman\Downloads\Nigerian Student's Year One Performance Survey(1-174).xlsx",index_col=0)
st.dataframe(df)
