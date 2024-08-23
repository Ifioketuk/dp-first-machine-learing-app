import streamlit as st
import pandas as pd
pd.plotting.register_matplotlib_converters()
import os
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



st.title('🤖 CGPA Prediction App')

st.write('This is an app used to predict the cgpa at the end of your first year based off academic performance at the end of high-school and study habits during the first year semesters.')
df= pd.read_excel(r"C:\Users\emman\Downloads\Nigerian Student's Year One Performance Survey(1-174).xlsx",index_col=0)
df
