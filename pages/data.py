import streamlit as st
import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from webdriver_manager.core.os_manager import ChromeType
from bs4 import BeautifulSoup
import time
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from scipy.special import softmax


def scrape_amazon_reviews(inputx):
    def web_driver():
        options = Options()
        options.add_argument("--verbose")
        options.add_argument('--no-sandbox')
        options.add_argument('--headless')
        options.add_argument('--disable-gpu')
        options.add_argument("--window-size=1920,1200")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.0.0 Safari/537.36")
        
        return webdriver.Chrome(
            service=Service(
                ChromeDriverManager(chrome_type=ChromeType.CHROMIUM).install()
            ),
            options=options,
        )

    def try_execute(func, *args, **kwargs):
        while True:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"An error occurred: {e}. Trying again...")
                time.sleep(2)
    
    driver = web_driver()
    driver.get('https://www.amazon.in/')
    driver.maximize_window()

    time.sleep(5)
    isearch = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.NAME, 'field-keywords')))
    sbutton = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.ID, 'nav-search-submit-button')))
    
    isearch.send_keys(inputx)
    time.sleep(1)
    sbutton.click()

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    button_element = soup.find('a', {'class': 'a-link-normal s-underline-text s-underline-link-text s-link-style a-text-normal'})
    if not button_element:
        print(html)
        driver.quit()
        raise ValueError("No product found on Amazon with the given name.")
    
    button_class = button_element['class']

    button_selector = f".{'.'.join(button_class)}"
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, button_selector)))
    element.click()
    st.write("Result Found")
    
    driver.switch_to.window(driver.window_handles[1])
    time.sleep(10)
    
    st.markdown(f"<a href='{driver.current_url}' target='_blank'>{'Product Page'}</a>", unsafe_allow_html=True)
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    details = soup.find_all('label', {'class': 'a-form-label'})
    details_list = [label.get_text(strip=True) for label in details]

    morereviewbutton1 = try_execute(WebDriverWait(driver, 20).until, EC.element_to_be_clickable((By.LINK_TEXT, "See more reviews")))
    morereviewbutton1.click()

    url1 = driver.current_url
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    def get_reviews_for_rating(driver, rating_text):
        return try_execute(_get_reviews_for_rating, driver, rating_text)
    
    def _get_reviews_for_rating(driver, rating_text):
        driver.refresh()
        rating_button = WebDriverWait(driver, 20).until(EC.element_to_be_clickable((By.LINK_TEXT, rating_text)))
        rating_button.click()
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        reviews = soup.find('div', {'data-hook': 'cr-filter-info-review-rating-count'}).text.strip()
        return reviews

    stars = {'1 star': None, '2 star': None, '3 star': None, '4 star': None, '5 star': None}
    
    for rating in stars.keys():
        reviews = get_reviews_for_rating(driver, rating)
        reviews = get_reviews_for_rating(driver, rating)
        stars[rating] = reviews
    
    updated_stars = {}
    
    for key, value in stars.items():
        if value is not None:
            try:
                rating, review = value.split('total ratings, ')
                rating = rating.strip().replace(',', '')
                review = review.split(' with reviews')[0].strip().replace(',', '')
                updated_stars[key] = {'ratings': int(rating), 'reviews': int(review)}
            except Exception as e:
                print(f"An error occurred while processing rating {key}: {e}")
        else:
            updated_stars[key] = {'ratings': 0, 'reviews': 0}
    
    all_reviews = pd.DataFrame()
    rating_to_filter = {
        '1 star': 'one_star',
        '2 star': 'two_star',
        '3 star': 'three_star',
        '4 star': 'four_star',
        '5 star': 'five_star'
    }
    
    def get_reviews(driver, rating_text, total_reviews, filters):
        return try_execute(_get_reviews, driver, rating_text, total_reviews, filters)
    
    def _get_reviews(driver, rating_text, total_reviews, filters):
        reviews_list = []
        collected_reviews = 0   
        page_number = 1

        while page_number <= 10:
            time.sleep(2)
            url = f"{url1}&pageNumber={page_number}&filterByStar={filters}"
            driver.get(url)
            rating_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.LINK_TEXT, rating_text)))
            rating_button.click()
            html = driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            reviews = soup.find_all('div', {'data-hook': 'review'})
            
            for review in reviews:  
                title = review.find('a', {'data-hook': 'review-title'}).text.strip() if review.find('a', {'data-hook': 'review-title'}) else ''
                rating = review.find('i', {'data-hook': 'review-star-rating'}).text.strip() if review.find('i', {'data-hook': 'review-star-rating'}) else ''
                date = review.find('span', {'data-hook': 'review-date'}).text.strip() if review.find('span', {'data-hook': 'review-date'}) else ''
                body = review.find('span', {'data-hook': 'review-body'}).text.strip() if review.find('span', {'data-hook': 'review-body'}) else ''
                details = review.find('a', {'data-hook': 'format-strip'}).text.strip() if review.find('a', {'data-hook': 'format-strip'}) else ''

                reviews_list.append({
                    'title': title,
                    'rating': rating,
                    'date': date,
                    'body': body,
                    'details': details
                })

                collected_reviews += 1
                print(f"Scraped {collected_reviews} reviews for {rating_text}")

                if collected_reviews >= total_reviews:
                    break

            page_number += 1

        return pd.DataFrame(reviews_list)

    for star, values in updated_stars.items():
        rating_text = star
        total_reviews = values['reviews']
        if total_reviews > 0:
            filters = rating_to_filter.get(rating_text, 'all_stars')
            print(f"Collecting {total_reviews} reviews for {rating_text} with filter {filters}...")
            reviews_df = get_reviews(driver, rating_text, total_reviews, filters)
            
            if reviews_df is not None:
                all_reviews = pd.concat([all_reviews, reviews_df], ignore_index=True)
    
    driver.quit()
    
    updated_stars_df = pd.DataFrame.from_dict(updated_stars, orient='index').reset_index().rename(columns={'index': 'star_rating'})
    
    return all_reviews, updated_stars_df, details_list




def process_details(df, det):
    def split_details(row, det):
        details_dict = {}
        for i, key in enumerate(det):
            start_idx = row.find(key) + len(key)
            if i + 1 < len(det):
                end_idx = row.find(det[i + 1])
            else:
                end_idx = len(row)
            details_dict[key[:-1].strip()] = row[start_idx:end_idx].strip()
        return details_dict
    
    details_df = df['details'].apply(lambda x: pd.Series(split_details(x, det)))
    df = df.join(details_df)
    df = df.drop(columns=['details'])
    return df

def clean_data(df, det):
    for col in ['title', 'date', 'body', 'rating', 'details']:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    df[['title', 'date', 'body']] = df[['title', 'date', 'body']].astype(str)

    string_to_remove = "The media could not be loaded."
    df = df.replace(string_to_remove, "", regex=True)
    df.replace({"": np.nan, "-": np.nan, "Null": np.nan}, inplace=True)

    if df['details'].isna().mean() >= 0.75:
        print("Dropping 'details' column as 75% or more of the values are empty.")
        df.drop(columns=['details'], inplace=True)
        df.dropna(axis=0, inplace=True)
    else:
        df.dropna(axis=0, inplace=True)
        df = process_details(df, det)
    
    #df.dropna(axis=0, inplace=True)
    df['title'] = df['title'].str.split('\n').str[1]
    df['date'] = df['date'].str.split('on ').str[1]
    df['date'] = pd.to_datetime(df['date'], format='%d %B %Y').dt.date

    try:
        df['rating'] = df['rating'].str.split(' ', expand=True)[0]
        df['rating'] = df['rating'].astype(float).astype(int)
    except KeyError:
        print("Error: 'rating' column split operation failed. Check if the data format is as expected.")

    return df

def translate_dataframe(df, columns, target_language='en'):
    from deep_translator import GoogleTranslator
    translator = GoogleTranslator(source='auto', target=target_language)
    
    total_cells = len(df.index) * len(columns)
    progress_bar = st.progress(0)
    progress = 0

    for column in columns:
        for index in df.index:
            if isinstance(df.at[index, column], str):
                df.at[index, column] = translator.translate(df.at[index, column])
            progress += 1
            progress_bar.progress(progress / total_cells)
    
    df.replace({"": np.nan, "-": np.nan, "null": np.nan}, inplace=True)
    df.dropna(axis=0, inplace=True)
    return df


def analyze_sentiment(df, text_column):
    # Nested function to preprocess text
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    # Load model and tokenizer
    MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Function to analyze sentiment of a single text
    def analyze_single_sentiment(text):
        text = preprocess(text)
        encoded_input = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        top_index = np.argmax(scores)
        top_label = config.id2label[top_index]
        return top_label

    # Apply sentiment analysis to the entire dataframe
    total_reviews = len(df)
    progress_bar = st.progress(0)
    progress = 0

    sentiments = []
    for review in df[text_column]:
        sentiment = analyze_single_sentiment(review)
        sentiments.append(sentiment)
        progress += 1
        progress_bar.progress(progress / total_reviews)

    df['sentiment'] = sentiments
    return df


# Streamlit dashboard
st.title("Amazon Reviews Scraper")

input_text = st.text_input("Enter the product name:", value="iphone")

if st.button("Scrape Reviews"):
    with st.spinner("Scraping reviews..."):
        try:
            # Scrape Amazon reviews
            reviews_df, ratings_df, details_list = scrape_amazon_reviews(input_text)
            st.success("Scraping completed!")

            # Display and save the raw dataframes
            st.write("Raw Reviews data:")
            st.dataframe(reviews_df)
            #reviews_df.to_excel("reviewdata.xlsx", index=False)
            #file = open('items.txt','w')
            #for det in details_list:
            #    file.write(det+"\n")
            #file.close()

            st.write("Raw Ratings data:")
            st.dataframe(ratings_df)
            #ratings_df.to_excel("ratingdata.xlsx", index=False)
            st.session_state['rating'] = ratings_df

            # Clean the reviews dataframe
            st.write("Cleaning data...")
            cleaned_reviews_df = clean_data(reviews_df, details_list)
            st.write("Cleaned Reviews data:")
            st.dataframe(cleaned_reviews_df)
            #cleaned_reviews_df.to_excel("cleaned_reviewdata.xlsx", index=False)

            # Translate the cleaned reviews dataframe
            st.write("Translating data...")
            cleaned_reviews_df = translate_dataframe(cleaned_reviews_df, ['title', 'body'])
            st.write("Translated Reviews data:")
            st.dataframe(cleaned_reviews_df)

            # Perform sentiment analysis on the cleaned reviews dataframe
            st.write("Performing Sentiment Analysis on data...")
            cleaned_reviews_df = analyze_sentiment(cleaned_reviews_df, 'body')
            st.dataframe(cleaned_reviews_df)
            #cleaned_reviews_df.to_excel("cleaned_reviewdata_with_sentiment.xlsx", index=False)
            st.session_state['review'] = cleaned_reviews_df

        except Exception as e:
            st.error(f"An error occurred: {e}")


