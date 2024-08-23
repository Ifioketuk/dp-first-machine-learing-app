import streamlit as st
import requests
import pandas as pd

st.title('🤖 CGPA Prediction App')

st.write('This is an app used to predict the cgpa at the end of your first year based off academic performance at the end of high-school and study habits during the first year semesters.')
df= pd.read_excel(r"C:\Users\emman\Downloads\Nigerian Student's Year One Performance Survey(1-174).xlsx",index_col=0)
df
