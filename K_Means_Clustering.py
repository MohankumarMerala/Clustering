#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pickle import dump, load

# For Suppressing warnings
import warnings
warnings.filterwarnings('ignore')

# For Hopkins Statistics
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan

# Feature Scaling
from sklearn.preprocessing import StandardScaler

# For K Means
from sklearn.cluster import KMeans


# In[2]:


import streamlit as st




# In[6]:


st.title('Model Deployment: Cluster Prediction')


Birthrate = st.number_input("Enter Birth Rate")

TaxPercentage = st.number_input("Enter Tax Percentage")

Co2emissions = st.number_input("Enter co2emissions")

EnergyUsage = st.number_input("Enter Energy Usage")

GDP = st.number_input("Enter GDP")

HealthExpGDP = st.number_input("Health Exp per GDP")

HealthExpCapita = st.number_input("Enter Health Exp/Capita")

InfantMortalityRate = st.number_input("Enter Infant Mortality Rate")

InternetUsage = st.number_input("Enter Internet Usage")

LifeExpectancyFemale = st.number_input("Enter Life Expectancy Female")

LifeExpectancymale = st.number_input("Enter Life Expectancy male")

MobilePhoneUsage = st.number_input("Enter Mobile Phone Usage")

Population0to14 = st.number_input("Enter Population 0-14")

Population15to64 = st.number_input("Enter Population 15-64")

Population65plus = st.number_input("Enter Population 65+")

PopulationTotal = st.number_input("Enter Population Total")

PopulationUrban = st.number_input("Enter Population Urban")

TourismInbound = st.number_input("Enter Tourism Inbound")

TourismOutbound = st.number_input("Enter Tourism Outbound")



loaded_model = load(open("final_model1.sav", 'rb'))

list1 = [Birthrate,TaxPercentage,Co2emissions,EnergyUsage,GDP,HealthExpGDP,HealthExpCapita,InfantMortalityRate, InternetUsage,LifeExpectancyFemale,LifeExpectancymale,MobilePhoneUsage,Population0to14,Population15to64,Population65plus,PopulationTotal,PopulationUrban,TourismInbound,TourismOutbound]
import numpy as np

# reshape the input data to a 2D array
list1 = np.array(list1).reshape(1, -1)

# predict the cluster label for the input data
result = loaded_model.predict(list1)

# display the predicted cluster label
submit = st.button('Submit')
if submit:
    if result==0:
        st.write("Under Developing")

    elif result==1:
        st.write("Developed Countries")

    else:
        st.write("Developing Countries")
