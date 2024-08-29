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
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import joblib

import time
# Load the trained model
model = joblib.load("model.json")
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
    age_in_year_one=st.number_input("age_in_year_one",format="%d")
    gender=st.select_slider('gender', options=['male','female'])
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
    grading_system = st.select_slider('grading system', options=[4, 5, 7, 10])

    
col1, col2= st.columns(2)
with col1:
    st.write("Give your ratings in the following")
    class_participation_rating = st.radio("class participation rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    class_attendance_rating=st.radio("class attendance rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    teaching_style_rating = st.radio("teaching style rating",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    extracurricular_participation = st.radio("extracurricular participation",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
with col2:
    morning_study = st.radio("morning study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    afternoon_study = st.radio("afternoon study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    evening_study = st.radio("evening study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    late_night_study= st.radio("late night study",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
    used_extra_study_materials=st.radio("use of extra study materials",('⭐', '⭐⭐', '⭐⭐⭐', '⭐⭐⭐⭐', '⭐⭐⭐⭐⭐'))
col1, col2= st.columns(2)
with col1:
    hours_per_day_personal_study=st.number_input('hours_per_day_personal_study',min_value=0,max_value=24,format="%d")
    days_per_week_reading=st.number_input('How many days of the week do you read',min_value=0,max_value=7,format="%d")
    study_mode= st.select_slider('study mode', options=['part time','full time'])
    taught_peers= st.select_slider('taught_peers',options=["Yes","No"])
with col2:
    extra_curricular= st.select_slider('Did you attend extra-curriculars?',options=["Yes","No"])
    allowance=st.select_slider('What was your monthly allowance in Year One?', options=["10k-20k", "30k-50k", "60k-100k", "above 100k"])
    courses_offered=st.number_input("Number of courses offered",format="%d")
studied_original_course= st.select_slider('Did you study your original course',options=["Yes","No"])

                                                                                        
                                                                                        
if st.button('Show CGPA'):                                                                                        
    data ={    
                                                                                            
        'jamb_score':jamb_score ,
        'english': english,
        'maths': maths ,
        'subject_3': subject_3,
        'subject_4': subject_4,
        'subject_5': subject_5,
        'age_in_year_one': age_in_year_one,
        'gender':gender ,
        'has_disability': has_disability,
        'Did you attend extra tutorials? ':extra_curricular ,
        'extracurricular_participation':extracurricular_participation,
        'class_attendance_rating': class_attendance_rating ,
        'class_participation_rating': class_participation_rating,
        'used_extra_study_materials': used_extra_study_materials,
        'morning_study':morning_study ,
        'afternoon_study':afternoon_study ,
        'evening_study':evening_study ,
        'late_night_study':late_night_study ,
        'days_per_week_reading': days_per_week_reading,
        'hours_per_day_personal_study':hours_per_day_personal_study ,
        'taught_peers': taught_peers,
        'courses_offered':courses_offered ,
        'times_fell_sick':times_fell_sick ,
        'study_mode': study_mode,
        'studied_original_course':studied_original_course ,
        'What was your monthly allowance in Year One?':allowance ,
        'teaching_style_rating': teaching_style_rating,
        'institution_type':institution_type ,
        'What year did you finish Year One?':year,
        'grading_system': grading_system}
    

    gpa_data_comp_col = pd.DataFrame([data])

    st.write("User Inputs DataFrame:")
    st.dataframe(gpa_data_comp_col)

    # Define grade features and mapping
    grade_features = ['maths', 'english', 'subject_3', 'subject_4', 'subject_5']
    grade_to_value = {'A': 4, 'B': 3, 'C': 2, 'D': 1, 'F': 0}
    
    # Map grades to numeric values
    for col in grade_features:
        gpa_data_comp_col[col] = gpa_data_comp_col[col].map(grade_to_value)
    
    # Calculate Combined Grade
    gpa_data_comp_col['Combined_grade'] = gpa_data_comp_col[grade_features].sum(axis=1)
    
    # Apply the condition for Combined Grade update
    gpa_data_comp_col.loc[
        (gpa_data_comp_col['maths'] < 1) | (gpa_data_comp_col['english'] < 1),
        'Combined_grade'
    ] = gpa_data_comp_col['Combined_grade'] / 3
    
    # Rank grades
    bins = [0, 10, 15, 20]  # Define bins
    labels = ['Rank 3', 'Rank 2', 'Rank 1']  # Define labels
    gpa_data_comp_col['grade_rank'] = pd.cut(
        gpa_data_comp_col['Combined_grade'],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True
    )
    
    # Load KMeans model
    loaded_kmeans = joblib.load('kmeans_model.pkl')
    gpa_data_comp_col['cluster'] = loaded_kmeans.predict(gpa_data_comp_col[grade_features + ['Combined_grade']])
    
    # Prepare the data for the main model
    features_to_drop = ['What year did you finish Year One?', 'english', 'maths', 'subject_3', 'subject_4', 'subject_5']
    Z = gpa_data_comp_col.drop(features_to_drop, axis=1)
    
    # Ensure categorical columns are encoded
    categorical_columns = [col for col in Z.columns if Z[col].dtype == 'object']
    Z[categorical_columns] = Z[categorical_columns].astype(str)
    
    # Apply ordinal encoding to categorical columns
    ordinal_enc = OrdinalEncoder()
    Z[categorical_columns] = ordinal_enc.fit_transform(Z[categorical_columns])
    
  
    
    # Ensure the features match the model training set
    expected_features = model.feature_names_in_
    Z = Z[expected_features]
    
    # Make predictions
    predictions = model.predict(Z)
    
    # Display results
    st.write("🤖 Your CGPA is:")
    st.write(predictions * gpa_data_comp_col["grading_system"].values[0])  # Scale 
        
        



