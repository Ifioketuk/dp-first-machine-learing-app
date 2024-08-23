import streamlit as st
import requests
import pandas as pd
url = "https://github.com/Ifioketuk/dp-first-machine-learing-app/blob/master/Nigerian%20Student's%20Year%20One%20Performance%20Survey(1-174).xlsx"

st.title('🤖 CGPA Prediction App')

st.write('This is an app used to predict the cgpa at the end of your first year based off academic performance at the end of high-school and study habits during the first year semesters.')
response= requests.get(url)

with open("your_file.py") as file:
      file.write(response.content)
