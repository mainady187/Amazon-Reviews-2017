import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import requests
from bs4 import BeautifulSoup
import time
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Live Amazon Sentiment Agent", layout="wide")

@st.cache_resource
def load_assets():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    final_model = joblib.load('sentiment_model.pkl')
    return tfidf, final_model

def load_and_prepare_database(file_path: str) -> pd.DataFrame:
    db_df = pd.read_csv(file_path)
    df = db_df.dropna(subset=['Review Text']).copy()
    
    link_col = [col for col in df.columns if col.lower() in ['pageurl', 'product link', 'link', 'url']]
    
    if link_col:
        df['Product Link Clean'] = df[link_col[0]].astype(str).str.strip().str.lower()
    else:
        df['Product Link Clean'] = ""
        
    return df

def scrape_live_reviews(product_url: str) -> list:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    if "/dp/" in product_url:
        product_url = product_url.replace("/dp/", "/product-reviews/")
        
    time.sleep(1)
    try:
        response = requests.get(product_url, headers=headers, timeout=12)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.content, "html.parser")
        review_elements = soup.find_all("span", {"data-hook": "review-body"})
        if not review_elements:
            review_elements = soup.select(".review-text-content")
        return [el.get_text().strip() for el in review_elements if el.get_text().strip()]
    except Exception:
        return []

def display_dashboard_metrics(total: int, pos: int, neg: int, pos_pct: float):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Extracted Reviews", total)
    col2.metric("Positive Sentiment ", f"{pos} ({pos_pct:.1f}%)")
    col3.metric("Negative Sentiment ", f"{neg} ({100 - pos_pct:.1f}%)")

def plot_visualizations(df_visual: pd.DataFrame, pos: int, neg: int):
    
    counts = df_visual['Predicted_Sentiment'].value_counts().reindex([0, 1], fill_value=0)
    bar_data = pd.DataFrame({
        'Sentiment': ['Negative', 'Positive'],
        'Count': [counts[0], counts[1]]
    })

    col4, col5 = st.columns(2)
    
    with col4:
        
        fig_bar = px.bar(
            bar_data, 
            x='Sentiment', 
            y='Count',
            title="Linear SVM Predicted Distribution",
            color='Sentiment',
            color_discrete_map={'Negative': '#ff00ff', 'Positive': '#00f2ff'} 
        )
        
        
        fig_bar.update_layout(
            plot_bgcolor='#0F172A',
            paper_bgcolor='#0F172A',
            title_font=dict(size=16, color='#00f2ff', family="Arial"),
            font=dict(color='#ffffff'),
            showlegend=False,
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)', title="Count"),
            xaxis=dict(title="Sentiment")
        )
        
        fig_bar.update_traces(texttemplate='%{y}', textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with col5:
        
        fig_pie = px.pie(
            names=['Positive', 'Negative'],
            values=[pos, neg],
            title="Sentiment Share Percentage",
            color=['Positive', 'Negative'],
            color_discrete_map={'Positive': '#00f2ff', 'Negative': '#ff00ff'}
        )
        
      
        fig_pie.update_layout(
            paper_bgcolor='#0F172A',
            title_font=dict(size=16, color='#00f2ff', family="Arial"),
            font=dict(color='#ffffff'),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
       
        fig_pie.update_traces(textinfo='percent+label', hole=0.3) 
        st.plotly_chart(fig_pie, use_container_width=True)


def main():
    st.title("Live Amazon Product Sentiment Agent")
    st.markdown("Enter an Amazon product link. The agent will check our secure database or fetch reviews in real-time.")
    
    try:
        df = load_and_prepare_database('cleaned_amazon_reviews.csv')
        text_vectorizer, model = load_assets()
    except FileNotFoundError as e:
        st.error(f"Missing essential asset files: {e.filename}. Please check your directory.")
        return

    with st.expander("View Available Sample Links in Dataset"):
        valid_links = df[df['Product Link Clean'] != '']['Product Link Clean'].unique()
        if len(valid_links) > 0:
            st.write(valid_links[:3])
        else:
            st.write("No explicit links column found in the cleaned file.")
        
    search_link = st.text_input("🔗Paste Amazon Product Link Here:", placeholder="https://www.amazon.com/dp/...")
    
    if search_link:
        cleaned_search_link = search_link.strip().lower()
        live_reviews = []
        is_from_db = False
        
        with st.spinner("🔍 Analysing link and searching in local secure database..."):
            if 'Product Link Clean' in df.columns and cleaned_search_link != "":
                db_match = df[df['Product Link Clean'] == cleaned_search_link]
                if db_match.empty:
                    db_match = df[df['Product Link Clean'].apply(lambda x: str(x) in cleaned_search_link or cleaned_search_link in str(x) if str(x) else False)]
                
                if not db_match.empty:
                    live_reviews = db_match['Review Text'].dropna().tolist()
                    is_from_db = True
                
        if not live_reviews:
            with st.spinner("Link not fully matched in database. Launching Live Web Scraper..."):
                live_reviews = scrape_live_reviews(search_link)
                
        if live_reviews:
            if is_from_db:
                st.success(f"Product identified in local Database! Loaded {len(live_reviews)} verified reviews instantly.")
            else:
                st.success(f"Live Scraper Bypass Success! Fetched {len(live_reviews)} real-time reviews.")
                
            product_df = pd.DataFrame({"Review Text": live_reviews})
            product_X = text_vectorizer.transform(product_df['Review Text'])
            product_preds = model.predict(product_X)
            product_df['Predicted_Sentiment'] = product_preds
            
            total_reviews = len(product_df)
            pos_reviews = np.sum(product_preds == 1)
            neg_reviews = np.sum(product_preds == 0)
            pos_percentage = (pos_reviews / total_reviews) * 100
            
            display_dashboard_metrics(total_reviews, pos_reviews, neg_reviews, pos_percentage)
            plot_visualizations(product_df, pos_reviews, neg_reviews)
            
            st.subheader("Raw Reviews & AI Sentiment Decisions")
            display_df = product_df.copy()
            display_df['Predicted_Sentiment'] = display_df['Predicted_Sentiment'].map({1: "Positive", 0: "Negative"})
            st.dataframe(display_df, use_container_width=True)
        else:
            st.error("Could not find this link in database, and Amazon blocked the live request. Please copy a link directly from the expander above!")

if __name__ == "__main__":
    main()