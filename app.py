import streamlit as st
from catboost import CatBoostRegressor
import pickle
import pandas as pd
import numpy as np
import torch
import ast

# Load model and objects
catboost_model = CatBoostRegressor().load_model('catboost_model.cbm')
scaler = pickle.load(open('scaler.pkl', 'rb'))
bert_pca = pickle.load(open('bert_pca.pkl', 'rb'))
img_pca = pickle.load(open('img_pca.pkl', 'rb'))

# Features used
features_to_scale = ['sentiment', 'readability', 'review_length', 'punctuation_count',
                     'verified_purchase', 'avg_quality_score', 'days_since_review',
                     'rating_number', 'average_rating', 'rating_deviation']

# Load data
df = pd.read_csv("sample_reviews.csv")

# Helper functions
def parse_embedding(embed, dim):
    try:
        if isinstance(embed, str):
            embed_str = embed.replace('e+', 'e').replace('[', '').replace(']', '').strip()
            embed = [float(x) for x in embed_str.split() if x != '...']
            embed = np.array(embed, dtype=np.float32)
        elif isinstance(embed, (list, np.ndarray, torch.Tensor)):
            embed = np.array(embed, dtype=np.float32)
        else:
            embed = np.zeros(dim, dtype=np.float32)
        if embed.shape[0] != dim:
            embed = np.zeros(dim, dtype=np.float32)
        if not np.all(np.isfinite(embed)):
            embed = np.zeros(dim, dtype=np.float32)
        return embed
    except:
        return np.zeros(dim, dtype=np.float32)

def prepare_features(df):
    bert_embeds = np.stack(df['bert_embedding'].values)
    img_embeds = np.stack(df['image_embedding'].values)
    other_feats = df[features_to_scale].values
    features = np.concatenate([bert_embeds, img_embeds, other_feats], axis=1)
    return features

def predict_review_helpfulness(review_data):
    review_df = pd.DataFrame([review_data])
    bert_dim, img_dim = 768, 512
    review_df['bert_embedding'] = review_df['bert_embedding'].apply(lambda x: parse_embedding(x, bert_dim))
    review_df['image_embedding'] = review_df['image_embedding'].apply(lambda x: parse_embedding(x, img_dim))
    review_df['bert_embedding'] = list(bert_pca.transform(np.stack(review_df['bert_embedding'].values)))
    review_df['image_embedding'] = list(img_pca.transform(np.stack(review_df['image_embedding'].values)))
    
    review_df['rating_deviation'] = review_df['rating_number'] - review_df['average_rating']
    for col in features_to_scale:
        review_df[col] = review_df[col].astype(float).fillna(0).replace([np.inf, -np.inf], 0)
    review_df[features_to_scale] = scaler.transform(review_df[features_to_scale])
    
    features = prepare_features(review_df)
    return catboost_model.predict(features)[0]

# Streamlit UI
st.title("🧠 Multimodal Amazon Review Helpfulness Predictor")

review_idx = st.selectbox("Select a Review", df.index)
review_data = df.iloc[review_idx].to_dict()

with st.expander("🔍 Input Features"):
    st.markdown("### 📝 Review Text")
    st.write(review_data.get("text", "Not available"))
    
    st.markdown("### 📊 Text Features")
    st.write({
        "Sentiment Score": review_data.get("sentiment"),
        "Readability": review_data.get("readability"),
        "Review Length": review_data.get("review_length"),
        "Punctuation Count": review_data.get("punctuation_count")
    })

    st.markdown("### 🧾 Metadata")
    st.write({
        "Verified Purchase": review_data.get("verified_purchase"),
        "Average Quality Score": review_data.get("avg_quality_score"),
        "Days Since Review": review_data.get("days_since_review"),
        "Total Number of Ratings": review_data.get("rating_number"),
        "Average Product Rating": review_data.get("average_rating"),
        "Product Category": review_data.get("main_category", "Unknown")
    })

    st.markdown("### 🖼️ Image Info")

    if isinstance(review_data.get("image_embedding"), str):
        st.write("Image embedding present.")
    else:
        st.write("No image embedding.")

    # Optional: display the actual image

    image_col = "image_urls" if "image_urls" in df.columns else None
    if image_col and pd.notna(review_data.get(image_col, None)):
        try:
            image_list = ast.literal_eval(review_data[image_col])  # convert string to actual list
            if isinstance(image_list, list) and len(image_list) > 0:
                st.image(image_list[0], caption="Product/Review Image", use_container_width=True)
            else:
                st.info("No valid image URL found.")
        except Exception as e:
            st.warning(f"Could not parse image URL: {e}")
    else:
        st.info("No image available to display.")

if st.button("🔮 Predict Helpfulness"):
    score = predict_review_helpfulness(review_data)
    percentile_score = score * 300
    st.success(f"Predicted Helpfulness Score: {percentile_score:.2f}")
