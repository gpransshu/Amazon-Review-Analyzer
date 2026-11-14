# Amazon Review Analyzer

A Streamlit web app that scrapes Amazon (amazon.in) product reviews and ratings, then analyzes and visualizes them — including per-review sentiment analysis with a Hugging Face transformer model.

## Overview

The app is a two-page workflow:

1. **Data page** — You enter a product name and the app:
   - Searches for the product on amazon.in with a headless Chromium browser (Selenium) and opens the first result.
   - Scrapes the total rating/review counts for each star level (1–5 stars) and collects the review title, rating, date, body, and format details for each star category (up to 10 result pages per category).
   - Cleans the data (drops nulls/placeholder text, parses dates, normalizes ratings, splits product "details" into separate columns).
   - Translates review titles and bodies to English (Google translation via `deep-translator`).
   - Runs sentiment analysis on each review body using the `cardiffnlp/twitter-roberta-base-sentiment-latest` model (labels: positive / neutral / negative).

2. **Visuals page** — Using the data stored in session state, it renders an interactive dashboard:
   - Gauge charts of ratings vs. reviews per star level, plus a coxcomb (polar bar) chart of the rating/review distribution.
   - Line chart of average sentiment over time (per month, with positive/neutral/negative annotations).
   - Word clouds of the top keywords per sentiment category, a keyword sunburst chart, and a 3D scatter of keyword frequency by sentiment (keywords are extracted with NLTK: punctuation removal, tokenization, POS-aware lemmatization, stop-word removal).
   - Bar charts of feature-level ("details") sentiment distribution and feature strength, when the product exposes such details.

Note: the Visuals page reads data from Streamlit session state, so you must complete a scrape on the Data page first — the Visuals page will error if no data has been collected in the current session.

## Tech Stack

- **Python** with **Streamlit** (multi-page app)
- **Selenium** + **webdriver-manager** driving headless **Chromium** for scraping; **BeautifulSoup** for HTML parsing
- **Hugging Face Transformers** (`cardiffnlp/twitter-roberta-base-sentiment-latest`) + **PyTorch** for sentiment analysis
- **pandas**, **NumPy** for data handling
- **deep-translator** (Google Translate) for review translation
- **NLTK** for text preprocessing (tokenization, lemmatization, stemming, stop words)
- **Plotly**, **Matplotlib**, **WordCloud** for the dashboard visuals

Dependencies are pinned in `requirements.txt` (`chromium` is listed in `packages.txt` as a required system package).

## Project Structure

```
.
├── main.py            # Streamlit entry point: home page with usage instructions
├── pages/
│   ├── data.py        # Data page: scraping, cleaning, translation, sentiment analysis
│   ├── visual.py      # Visuals page: interactive charts and word clouds
│   └── 1.txt          # Empty placeholder file
├── packages.txt       # System package notes (chromium)
└── requirements.txt   # Pinned Python dependencies
```

## How to Run

1. Install a Chromium browser and the Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the app:

   ```bash
   streamlit run main.py
   ```

3. In the app:
   - Open the **Data** page, enter a product name (e.g. `iphone`), and click **Scrape Reviews**. Wait for scraping, cleaning, translation, and sentiment analysis to finish.
   - Open the **Visuals** page to view the dashboard for the collected product.

Notes:

- Scraping is aimed at **amazon.in**; if Amazon's markup changes, the CSS selectors and link texts in `pages/data.py` may need updating.
- The browser runs headless with `--no-sandbox`; `webdriver-manager` downloads the matching driver automatically.
- Required NLTK data (`punkt`, `stopwords`, `wordnet`, etc.) and the sentiment model are downloaded automatically on first run.
