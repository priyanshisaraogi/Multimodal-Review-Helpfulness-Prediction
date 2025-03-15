import pandas as pd
import numpy as np
import nltk
import json
import re
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from datetime import datetime

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the User Reviews dataset
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True)
df_reviews = pd.DataFrame(dataset["full"])

# Load the Item Metadata dataset
dataset_meta = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_meta_All_Beauty", split="full", trust_remote_code=True)
df_meta = pd.DataFrame(dataset_meta)

### Step 1: Handle Missing Values ###
df_reviews.dropna(subset=['text'], inplace=True)  # Drop reviews with missing text
df_reviews.fillna({'helpful_vote': 0}, inplace=True)  # Fill missing helpful votes with 0
df_meta['price'] = df_meta['price'].replace('None', np.nan).astype(float)  # Convert price to numeric

### Step 2: Preprocess Review Text ###
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    words = word_tokenize(text)  # Tokenize words
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize
    return " ".join(words)

df_reviews['cleaned_text'] = df_reviews['text'].apply(clean_text)
df_reviews['review_length'] = df_reviews['cleaned_text'].apply(lambda x: len(x.split()))  # Compute review length

### Step 3: Normalize Helpfulness Score ###
df_reviews['helpfulness_score'] = df_reviews['helpful_vote'] / (df_reviews['helpful_vote'].max() + 1)  # Normalize between 0-1

### Step 4: Process Images ###
df_reviews['has_image'] = df_reviews['images'].apply(lambda x: 1 if len(x) > 0 else 0)

### Step 5: Convert Timestamp ###
df_reviews['review_date'] = df_reviews['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x / 1000).strftime('%Y-%m-%d'))

### Step 6: Process Metadata ###
df_meta['has_product_image'] = df_meta['images'].apply(lambda x: 1 if x and any(x['hi_res']) else 0)
df_meta['num_features'] = df_meta['features'].apply(lambda x: len(x) if isinstance(x, list) else 0)

# Convert JSON string 'details' column into dictionary
def extract_details(details):
    try:
        details_dict = json.loads(details.replace("'", '"')) if isinstance(details, str) else {}
        return details_dict.get("Package Dimensions", "Unknown")
    except:
        return "Unknown"

df_meta['product_dimensions'] = df_meta['details'].apply(extract_details)

### Step 7: Merge Review & Metadata Datasets ###
df_merged = pd.merge(df_reviews, df_meta[['parent_asin', 'average_rating', 'rating_number', 'price', 'has_product_image']], on="parent_asin", how="left")

### Step 8: Train-Test Split ###
train_df, test_df = train_test_split(df_merged, test_size=0.2, stratify=df_merged['helpfulness_score'] > 0.5, random_state=42)

print("Training Data Shape:", train_df.shape)
print("Test Data Shape:", test_df.shape)

# Save cleaned data for model training
train_df.to_csv("cleaned_train_data.csv", index=False)
test_df.to_csv("cleaned_test_data.csv", index=False)

print("Data cleaning complete. Train and Test datasets saved.")
