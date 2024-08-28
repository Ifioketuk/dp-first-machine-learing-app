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
from pickle import load
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
    age_in_year_one=st.number_input("age_in_year_one")
    gender=st.select_slider('gender', options=['Yes','No'])
    has_disability=st.select_slider('has_disability', options=['Yes','No'])
    times_fell_sick=st.text_input('times_fell_sick')
    institution_type=st.select_slider('institution_type', options=['public','private'])
    year=st.text_input('What year did you finish Year One')
with col2:
    jamb_score=st.number_input("Enter your JAMB score(1-400)",min_value=0,max_value=400,format="%d")
    st.write("GRADES(A-F) in the following:")
    english=st.select_slider('english', options=['A','B','C','D','F'])
    maths= st.select_slider('maths', options=['A','B','C','D','F'])
    subject_3 =st.select_slider('subject_3', options=['A','B','C','D','F']) 
    subject_4=st.select_slider('subject_4', options=['A','B','C','D','F'])
    subject_5=st.select_slider('subject_5', options=['A','B','C','D','F'])
    grading_system = st.select_slider('grading_system', options=[4, 5, 7, 10])

    
col1, col2 = st.columns(2)
with col1:
    st.write("Give your ratings in the following")
    class_participation_rating = st.radio("class_attendance_rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    teaching_style_rating = st.radio("teaching_style_rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    extracurricular_participation = st.radio("extracurricular_participation",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    morning_study = st.radio("morning_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    afternoon_study = st.radio("afternoon_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    evening_study = st.radio("evening_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    late_night_study= st.radio("late_night_study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    allowance=st.select_slider('grading_system', options=["10k-20k", "30k-50k", "60k-100k", "above 100k"])
with col2:
    hours_per_day_personal_study=st.number_input('hours_per_day_personal_study',min_value=0,max_value=24,format="%d")
    days_per_week_reading=st.number_input('days_per_week_reading',min_value=0,max_value=7,format="%d")
    

    

data = {
    'jamb_score': ,
    'english': ,
    'maths': ,
    'subject_3': ,
    'subject_4': ,
    'subject_5': ,
    'age_in_year_one': ,
    'gender': ,
    'has_disability': ,
    'Did you attend extra tutorials?': ,
    'extracurricular_participation':,
    'class_attendance_rating': ,
    'class_participation_rating': ,
    'used_extra_study_materials': ,
    'morning_study': ,
    'afternoon_study': ,
    'evening_study': ,
    'late_night_study': ,
    'days_per_week_reading': ,
    'hours_per_day_personal_study': ,
    'taught_peers': ,
    'courses_offered': ,
    'times_fell_sick': ,
    'study_mode': ,
    'studied_original_course': ,
    'What was your monthly allowance in Year One?': ,
    'teaching_style_rating': ,
    'institution_type': ,
    'What year did you finish Year One?':,
    'grading_system': 
}
model=load("model1.pkl",'rb')    






