import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from bs4 import BeautifulSoup
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import string

st.title("Amazon")

st.write("This project provides a detailed analysis of reviews and ratings for the product search input")

st.write("Step-by-Step Instructions")

st.write("1. **Go to the Data Page**")
st.write("   - Navigate to the Data page in the application to start the analysis process.")

st.write("2. **Search for the Product**")
st.write("   - Enter the name of the product you want to analyze in the search bar and initiate the search.")

st.write("3. **Wait for Data Collection**")
st.write("   - Allow time for the application to collect all relevant reviews and ratings data from Amazon.")
st.write("   - You can scroll down to monitor the progress of the data collection.")

st.write("4. **Go to the Visuals Page**")
st.write("   - Once data collection is complete, navigate to the Visuals page.")
st.write("   - Here, you can view detailed visualizations and analyses of the collected data.")
