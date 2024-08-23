import streamlit as st
import requests
import pandas as pd

# Define the URL of the Excel file on GitHub
url = "https://raw.githubusercontent.com/Ifioketuk/dp-first-machine-learing-app/main/Nigerian%20Student's%20Year%20One%20Performance%20Survey(1-174).xlsx"

# Fetch the file from GitHub
response = requests.get(url)

# Save the file locally
with open("Nigerian_Students_Year_One_Performance_Survey.xlsx", "wb") as file:
    file.write(response.content)

# Title and description for the app
st.title('🤖 CGPA Prediction App')
st.write('This is an app used to predict the CGPA at the end of your first year based on academic performance at the end of high school and study habits during the first year semesters.')

# Read the Excel file using pandas
df = pd.read_excel("Nigerian_Students_Year_One_Performance_Survey.xlsx", index_col=0)

# Display the dataframe in Streamlit
st.dataframe(df)
