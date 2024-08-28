import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

title=st.empty()
title.title('PRACTICE MODEL')

# Create placeholders for the progress bar and iteration text
latest_iteration = st.empty()
bar = st.empty()

# Progress bar setup
bar.progress(0)

# Long computation simulation
for i in range(100):
    # Update the progress bar with each iteration
    latest_iteration.text(f'loading {i+1}%')
    bar.progress(i + 1)
    time.sleep(0.01)

# Clear the progress bar and iteration text after completion
latest_iteration.empty()
bar.empty()

# Display the final message
exitstatement=st.empty()
exitstatement.write('...and now we\'re done!')
time.sleep(0.001)

#let both statements disappear
latest_iteration.empty()
bar.empty()
exitstatement.empty()
title.empty()
#actual title
st.title('🤖 CGPA Prediction App')
st.write('This is an app used to predict the CGPA at the end of your first year based on academic performance at the end of high school and study habits during the first year semesters.')

st.write("Fill in the following features to get your possible GPA")
col1, col2 = st.columns(2)
with col1:
    st.text_input("age_in_year_one")
    st.text_input("gender")
    st.text_input('has_disability')
    st.text_input('times_fell_sick')
    st.text_input("institution_type")
    st.text_input('What year did you finish Year One')
with col2:
    st.text_input("Enter your JAMB score(1-400)")
    st.write("GRADES(A-F) in the following:")
    st.text_input('english')
    st.text_input('maths')
    st.text_input('subject_3')  
    st.text_input('subject_4')
    st.text_input('subject_5')
st.write("Give your ratings in the following")
class_participation_rating = st.radio("class_attendance_rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
teaching_style_rating = st.radio("teaching_style_rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
extracurricular_participation = st.radio("extracurricular_participation",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
morning_study = st.radio("morning_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
afternoon_study = st.radio("afternoon_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
evening_study = st.radio("evening_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))

st.text_input("days_per_week_reading(1-7)")
st.text_input("monthly allowance in Year One")



    
