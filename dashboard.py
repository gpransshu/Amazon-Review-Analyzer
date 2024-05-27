import streamlit as st
import pandas as pd
import plotly.express as px

# Display an animation
animation_url = "Vanilla@1x-1.1s-277px-250px.gif"
st.image(animation_url, caption='Animation', use_column_width=True)

# Load the Excel file into a DataFrame
df = pd.read_excel('Dataset.xlsx')

# Display the DataFrame in Streamlit
st.write(df)

# Sample dataset
data = pd.DataFrame({
    'Rating': [5, 4, 3, 2, 1],
    'Count': [100, 200, 150, 50, 20]
})

# Streamlit app
st.title('Amazon Review Analyzer')

# Display the dataset
st.write('Sample Dataset Visual:')
st.write(data)


# Create a bar chart using Plotly
fig = px.bar(df['rating'].value_counts(),  x=df['rating'].value_counts().index, y=df['rating'].value_counts(), title='Rating Distribution')
st.plotly_chart(fig)

