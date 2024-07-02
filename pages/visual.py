import streamlit as st
st.set_page_config(layout="wide")  # Set wide layout for better visualization

import pandas as pd
import numpy as np

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

# Create a Streamlit app


# Download necessary NLTK resources (do this only once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Define stopwords list (feel free to modify)
stop_words = stopwords.words('english')

#df = pd.read_excel("cleaned_reviewdata_with_sentiment.xlsx")
df = st.session_state['review']
dfcopy = df.copy()
#df1 = pd.read_excel("ratingdata.xlsx")
df1 = st.session_state['rating']

# Streamlit dashboard
st.title("Amazon Reviews Scraper")

def plot_average_sentiment(df):
    # Convert 'date' column to datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Group by month and year directly from 'date' column
    df['month_year'] = df['date'].dt.to_period('M').astype(str)

    # Map sentiment to numeric values for averaging
    sentiment_mapping = {'negative': 1, 'neutral': 0, 'positive': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)

    # Calculate average sentiment score per month
    monthly_sentiment = df.groupby('month_year')['sentiment_score'].mean().reset_index()

    # Determine sentiment emojis based on sentiment score nearest values
    sentiment_emojis = {
        'positive': '😊',   # smiling face emoji
        'negative': '😔',   # pensive face emoji
        'neutral': '😐'     # neutral face emoji
    }

    # Function to map sentiment score to emoji
    def map_sentiment_to_emoji(score):
        if score > 0.5:
            return sentiment_emojis['positive']
        elif score < -0.5:
            return sentiment_emojis['negative']
        else:
            return sentiment_emojis['neutral']

    # Apply mapping function to create a new column 'emoji'
    monthly_sentiment['emoji'] = monthly_sentiment['sentiment_score'].apply(map_sentiment_to_emoji)

    # Create Plotly Express figure
    fig = px.line(monthly_sentiment, x='month_year', y='sentiment_score', 
                  title='Sentiment Towards Product Over Time',
                  labels={'month_year': 'Month Year', 'sentiment_score': 'Sentiment'},
                  line_shape='linear', render_mode='svg')

    # Add emojis to the plot
    for i, row in monthly_sentiment.iterrows():
        fig.add_annotation(x=row['month_year'], y=row['sentiment_score'], 
                           text=row['emoji'], showarrow=False, font_size=20)

    # Customize x-axis date format
    fig.update_xaxes(
        tickangle=45,
        title_text='Date'
    )

        # Customize y-axis labels to sentiment categories and remove grid lines
    fig.update_yaxes(
        tickvals=[-1, 0, 1],
        ticktext=['Negative', 'Neutral', 'Positive'],
        showgrid=False,
        title_text='Sentiment'
    )

    return fig



def create_gauge_chart(df, star_index):
    star_rating = df['star_rating'][star_index]
    reviews = df['reviews'][star_index]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=reviews,  # Reviews for the specified star rating
        title={'text': f"{star_rating} Reviews", 'font': {'size': 24}},
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': df['reviews'].mean(), 'increasing': {'color': "orange"}},
        gauge={
            'axis': {'range': [None, df['reviews'].max()], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'borderwidth': 2,
            'bordercolor': "lightgray",
            'steps': [
                {'range': [0, df['reviews'].max() * 0.5], 'color': 'orange'},
                {'range': [df['reviews'].max() * 0.5, df['reviews'].max()], 'color': 'darkgray'}
            ],
            'threshold': {
                'line': {'color': "orange", 'width': 4},
                'thickness': 0.75,
                'value': reviews
            }
        }
    ))

    fig.update_layout(font={'color': "lightgray", 'family': "Arial"})

    return fig


# Function to preprocess text (punctuation removal, lemmatization, stopword removal, optional stemming)
def preprocess_text(text, stemming=False):
  lemmatizer = WordNetLemmatizer()
  stemmer = PorterStemmer()  # For stemming (optional)

  # Remove punctuation
  text_without_punct = ''.join(char for char in text if char not in string.punctuation)

  # Tokenize text
  tokens = nltk.word_tokenize(text_without_punct.lower())

  # Remove stop words
  filtered_tokens = [token for token in tokens if token not in stop_words]

  # Lemmatization
  lemmas = [lemmatizer.lemmatize(token) for token in filtered_tokens]

  # Stemming (optional)
  if stemming:
    stemmed_words = [stemmer.stem(word) for word in lemmas]
    return stemmed_words
  else:
    return lemmas

# Apply preprocessing to each review in the 'body' column
df['body_processed'] = df['body'].apply(preprocess_text)

def extract_top_keywords(df):
    import re
    from collections import defaultdict, Counter
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Ensure you have the required NLTK data
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)

    # Define stopwords list (feel free to modify)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        """Convert POS tag to a format recognized by the WordNet lemmatizer."""
        if tag.startswith('J'):
            return nltk.corpus.wordnet.ADJ
        elif tag.startswith('V'):
            return nltk.corpus.wordnet.VERB
        elif tag.startswith('N'):
            return nltk.corpus.wordnet.NOUN
        elif tag.startswith('R'):
            return nltk.corpus.wordnet.ADV
        else:
            return nltk.corpus.wordnet.NOUN

    def preprocess_text(review):
        """
        Combine sentences, remove punctuation, lemmatize tokens, remove stop words, and filter out single alphabets.
        """
        combined_review = ' '.join(review)  # Combine sentences into one
        # Remove punctuation and convert to lower case
        cleaned_tokens = [re.sub(r'[^\w\s]', '', token).lower() for token in word_tokenize(combined_review)]
        # Get POS tags for lemmatization
        pos_tags = nltk.pos_tag(cleaned_tokens)
        # Lemmatize tokens with their POS tags
        lemmatized_tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
        # Remove single alphabets and stop words
        cleaned_tokens = [token for token in lemmatized_tokens if len(token) > 1 and token not in stop_words]
        return cleaned_tokens

    # Group DataFrame by sentiment and combine body into one sentence per sentiment
    sentiment_groups = df.groupby('sentiment')['body_processed'].apply(lambda x: [' '.join(review) for review in x])

    all_sentiment_words = defaultdict(list)

    for sentiment, processed_reviews in sentiment_groups.items():
        # Preprocess and tokenize each combined review
        processed_tokens = preprocess_text(processed_reviews)
        # Count word frequencies
        word_counts = Counter(processed_tokens)
        # Store top 25 frequent words for each sentiment category
        top_keywords = word_counts.most_common(50)
        all_sentiment_words[sentiment] = top_keywords

    return all_sentiment_words

top_keywords = extract_top_keywords(df)
print(top_keywords)
# Function to generate and display word cloud

def filter_top_keywords(top_keywords, top_n=10):

    top_keywords_filtered = {}
    
    for sentiment, keywords in top_keywords.items():
        # Sort keywords by value (frequency) in descending order
        keywords_sorted = sorted(keywords, key=lambda x: x[1], reverse=True)
        # Take top N keywords (if available)
        top_keywords_filtered[sentiment] = keywords_sorted[:top_n]
    
    return top_keywords_filtered

def generate_word_cloud(sentiment, keywords):
    # Generate the word cloud with transparent background
    wordcloud = WordCloud(width=800, height=400, background_color=None, mode='RGBA').generate_from_frequencies(dict(keywords))
    
    # Create a figure
    fig, ax = plt.subplots()
    plt.figure(figsize=(6, 4))
    
    
    # Display the word cloud on the figure
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
    ax.axis('off')
    
    # Return the figure
    return fig
# Assuming 'top_keywords' is a dictionary with most frequent words per sentiment

def create_coxcomb_chart(df):
    # Extracting data from the DataFrame
    star_rating = df['star_rating'].tolist()
    ratings = df['ratings'].tolist()
    reviews = df['reviews'].tolist()

    # Create the Coxcomb chart
    fig = go.Figure()

    # Add ratings data to the chart
    fig.add_trace(go.Barpolar(
        r=ratings,
        theta=star_rating,
        name='Ratings',
        marker_color=['#FF6347', '#FFA07A', '#FFD700', '#ADFF2F', '#32CD32'],
        hoverinfo='none',  # Disable default hoverinfo
        hovertemplate='<b>%{theta}</b><br><i>Ratings</i>: %{r}<extra></extra>'
    ))

    # Add reviews data to the chart
    fig.add_trace(go.Barpolar(
        r=reviews,
        theta=star_rating,
        name='Reviews',
        marker_color=['#FF4500', '#FF8C00', '#FFD700', '#9ACD32', '#228B22'],
        hoverinfo='none',  # Disable default hoverinfo
        hovertemplate='<b>%{theta}</b><br><i>Reviews</i>: %{r}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title='Ratings and Reviews Distribution',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(ratings) + 10]
            )
        ),
        showlegend=False
    )

    # Show the chart
    return fig


def create_keyword_sunburst(top_keywords):
    # Create a list of dictionaries from the filtered data
    data = []
    
    for sentiment, keywords in top_keywords_filtered.items():
        for word, value in keywords:
            data.append({'keyword': word, 'parent': sentiment, 'value': value})

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    # Define a custom color map
    color_map = {
        'positive': 'green',
        'neutral': 'lightblue',
        'negative': 'red'
    }

    # Create the sunburst chart
    fig = px.sunburst(df, path=['parent', 'keyword'], values='value',
                      title="Top 10 Keyword Frequency by Sentiment",
                      color='parent',
                      color_discrete_map=color_map)

    return fig



def create_keyword_scatter_3d(top_keywords):
    # Create a list of dictionaries from the given data
    data = []

    for sentiment, keywords in top_keywords.items():
        for word, value in keywords:
            data.append({'keyword': word, 'sentiment': sentiment, 'value': value})

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Assign numerical values to sentiments for plotting
    sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['sentiment_num'] = df['sentiment'].map(sentiment_mapping)

    # Create the 3D scatter plot with custom colors
    fig = px.scatter_3d(
        df,
        x='keyword',
        y='sentiment_num',
        z='value',
        color='sentiment',
        title="3D Scatter Plot of Keywords by Sentiment",
        labels={'sentiment_num': 'Sentiment'},
        color_discrete_map={'negative': 'red', 'neutral': 'lightblue', 'positive': 'green'}
    )

    return fig


def bar(df):
    body_x = df.columns.get_loc("body")
    xaxis = df.iloc[:, body_x+1:].copy()

    def g(df, col):
        sentiment_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        df['sentiment_score'] = df['sentiment'].map(sentiment_mapping)
        return (df.groupby(col)['sentiment_score']
            .mean()
            .reset_index()
            .fillna(0))

    def plot(df):
        fig = make_subplots(rows=1, cols=len(list(df.columns))-1, shared_yaxes=True)

        # Loop through columns (assuming medals are in separate columns)
        for i, col in enumerate(df.columns[:-1]):
            a = g(xaxis.copy(),col)
            fig.add_trace(
            go.Bar(
                x=a[a.columns[0]].tolist(),
                y=a[a.columns[1]].tolist(),
                name=col.capitalize(),
                marker=dict(
                    color='orange',  # Set bar color
                    line=dict(color='white', width=2)  # Set frame properties
                ),   # Set color based on 'colour' column
            ),
            1,
            i + 1
            )
        #fig.update_layout(yaxis_range=(-1, 1))
        fig.update_layout(
        yaxis=dict(
            range=(-1, 1),  # Set y-axis range
            tickvals=[-1, 0, 1],  # Specify tick values for labels
            ticktext=["Negative", "Neutral", "Positive"],
            # Set custom labels for ticks
        ),
            title_text="Sentiment Distribution of Features",
            bargap=0.1
        )
        
        return fig
    fig = plot(xaxis)
    return fig

def bar1(df):
    body_x = df.columns.get_loc("body")
    xaxis = df.iloc[:, body_x+1:].copy()
    print(len(list(xaxis.columns)))

    def g(df, col):
        return df[col].value_counts()

    def plot(df):
        fig = make_subplots(rows=1, cols=len(list(df.columns))-1, shared_yaxes=True)

        # Loop through columns (assuming medals are in separate columns)
        for i, col in enumerate(df.columns[:-1]):
            a = g(xaxis.copy(),col)
            fig.add_trace(
            go.Bar(
                x=a.index.tolist(),
                y=a.values.tolist(),
                name=col.capitalize(),
                marker=dict(
                    color='orange',  # Set bar color
                    line=dict(color='white', width=2)  # Set frame properties
                ),   # Set color based on 'colour' column
            ),
            1,
            i + 1
            )
        #fig.update_layout(yaxis_range=(-1, 1))
        fig.update_layout(
            title_text="Features Strength Distribution",
            bargap=0.1
        )
        
        return fig
    fig = plot(xaxis)
    return fig


#Plotting

# First row: All five gauge charts
cols = st.columns([3,1])
with cols[0]:

#st.subheader('Gauge Charts')
    cols_gauges = st.columns(5)  # Adjust based on the number of gauges
    for i in range(5):
        with cols_gauges[i]:
            fig_gauge = create_gauge_chart(df1, i)
            st.plotly_chart(fig_gauge, use_container_width=True)

# Second row: Coxcomb chart, sunburst chart, and three word clouds
#st.subheader('Coxcomb Chart and Word Clouds')
    cols_row2 = st.columns(2)  # Adjust based on the number of charts and word clouds
    with cols_row2[0]:
        #st.subheader('Coxcomb Chart')
        fig_coxcomb = create_coxcomb_chart(df1)
        st.plotly_chart(fig_coxcomb, use_container_width=True)
    with cols_row2[1]:
        #st.subheader('Keyword Sunburst')
        top_keywords_filtered = filter_top_keywords(top_keywords, top_n=10)
        fig_sunburst = create_keyword_sunburst(top_keywords_filtered)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    fig_scatter_3d = create_keyword_scatter_3d(top_keywords_filtered)
    st.plotly_chart(fig_scatter_3d, use_container_width=True)



with cols[1]:
#st.subheader('Word Clouds')
    sentiments = ['positive', 'neutral', 'negative']
    for sentiment in sentiments:
            plt_wordcloud = generate_word_cloud(sentiment, top_keywords[sentiment])
            st.pyplot(plt_wordcloud, use_container_width=True)

    fig_bar1 = bar(dfcopy)
    st.plotly_chart(fig_bar1, use_container_width=True)

    fig_bar2 = bar1(dfcopy)
    st.plotly_chart(fig_bar2, use_container_width=True)


cols1 = st.columns(1)
with cols1[0]:
    fig_avg = plot_average_sentiment(df)
    st.plotly_chart(fig_avg, use_container_width=True)    
