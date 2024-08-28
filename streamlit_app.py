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

st.title('PRACTICE MODEL')

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
    time.sleep(0.1)

# Clear the progress bar and iteration text after completion
latest_iteration.empty()
bar.empty()

# Display the final message
st.write('...and now we\'re done!')
st.title('🤖 CGPA Prediction App')
st.write('This is an app used to predict the CGPA at the end of your first year based on academic performance at the end of high school and study habits during the first year semesters.')

# File uploader widget
uploaded_file = st.file_uploader("Nigerian Student's Year One Performance Survey(1-174).xlsx", type="xlsx")

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file, engine='openpyxl')

    new_column_names = {
        'ID': 'id',
        'Start time': 'start_time',
        'Completion time': 'completion_time',
        'Email': 'email',
        'Name': 'name',
        'Last modified time': 'last_modified_time',
        'Jamb score': 'jamb_score',
        'English': 'english',
        'Maths': 'maths',
        'Subject 3': 'subject_3',
        'Subject 4': 'subject_4',
        'Subject 5': 'subject_5',
        'What was your age in Year One': 'age_in_year_one',
        'Gender': 'gender',
        'Do you have a disability?': 'has_disability',
        'Did you attend extra tutorials? ': 'attended_tutorials',
        'How would you rate your participation in extracurricular activities (tech, music, partying, fellowship, etc.) in Year One?': 'extracurricular_participation',
        'How would you rate your class attendance in Year One': 'class_attendance_rating',
        'How well did you participate in class activities (Assignments, Asking and Answering Questions, Writing Notes....)': 'class_participation_rating',
        'Rate\xa0your use of extra materials for study in Year One (Youtube, Other books, others).': 'used_extra_study_materials',
        'Morning': 'morning_study',
        'Afternoon': 'afternoon_study',
        'Evening': 'evening_study',
        'Late Night': 'late_night_study',
        'How many days per week did you do reading on average in Year One?': 'days_per_week_reading',
        'On average, How many hours per day was used for personal study in Year One': 'hours_per_day_personal_study',
        'Did you teach your peers in Year One': 'taught_peers',
        'How many courses did you offer in Year One?': 'courses_offered',
        'Did you fall sick in Year One? if yes, How many times do you remember (0 if none)': 'times_fell_sick',
        'What was your study mode in Year 1': 'study_mode',
        'Did you study the course your originally applied for?': 'studied_original_course',
        'Rate your financial status in Year One': 'financial_status_rating',
        'Rate the teaching style / method of the lectures received in Year One': 'teaching_style_rating',
        'What type of higher institution did you attend in Year One\n': 'institution_type',
        'What was your CGPA in Year One?': 'cgpa_year_one',
        'What grading system does your school use ( if others, type numbers only)': 'grading_system'
    }
    df.rename(columns=new_column_names, inplace=True)
    columns_to_drop = ['start_time', 'completion_time', 'email', 'name', 'last_modified_time']
    df.drop(columns=columns_to_drop, inplace=True)

    gpa_data = df.copy()
    gpa_data_comp_col = gpa_data[gpa_data['cgpa_year_one'] != 'no idea'].copy()
    gpa_data_comp_col['cgpa_year_one'] = gpa_data_comp_col['cgpa_year_one'].str.replace('.o', '.0', regex=False)

    # Convert 'cgpa_year_one' to numeric, coercing errors to NaN
    gpa_data_comp_col['cgpa_year_one'] = pd.to_numeric(gpa_data_comp_col['cgpa_year_one'], errors='coerce')

    # Create KDE plot
    fig, ax = plt.subplots()
    sns.kdeplot(data=gpa_data_comp_col, x='times_fell_sick', hue='gender', fill=True, ax=ax)

    # Display the plot in Streamlit
    st.pyplot(fig)
else:
    st.write("Upload an Excel file to view the data and plots.")

