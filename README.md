# Multimodal Review Helpfulness Prediction

A comprehensive machine learning system that predicts the helpfulness of Amazon product reviews using multimodal data (text, images, and metadata). This project combines natural language processing, computer vision, and traditional machine learning to provide accurate helpfulness predictions.

## Project Overview

This system analyzes Amazon beauty product reviews and predicts how helpful they will be to other customers. By leveraging multiple modalities of data, the model achieves superior performance compared to text-only approaches.

🌐 **[Try the Live Demo](https://multimodal-review-prediction.streamlit.app/)** - Experience the model in action!

### Key Features

- **Text Analysis**: BERT embeddings, sentiment analysis, readability scores
- **Image Processing**: Deep learning-based image feature extraction  
- **Metadata Integration**: Review ratings, purchase verification, temporal features
- **Interactive Web App**: Streamlit-based interface for real-time predictions

## Architecture

The system uses a multi-stage pipeline:

1. **Data Preprocessing**: Text cleaning, image processing, feature engineering
2. **Feature Extraction**: 
   - BERT embeddings for text semantic representation
   - ResNet-based image embeddings
   - Handcrafted text features (sentiment, readability, length)
   - Metadata features (ratings, purchase verification, etc.)
3. **Dimensionality Reduction**: PCA for both text and image embeddings
4. **Model Training**: CatBoost regressor for final prediction
5. **Deployment**: Streamlit web application

## Project Structure

```
├── app.py                      # Streamlit web application
├── data_clean.py              # Data preprocessing and cleaning
├── data_preprocessing.ipynb   # Feature engineering notebook
├── final.ipynb               # Model training and evaluation
├── requirements.txt          # Python dependencies
├── sample_reviews.csv        # Sample dataset for demo
├── catboost_model.cbm       # Trained CatBoost model
├── bert_pca.pkl             # PCA model for BERT embeddings
├── img_pca.pkl              # PCA model for image embeddings
└── scaler.pkl               # Feature scaler
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Multimodal-Review-Helpfulness-Prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK data** (if running data preprocessing)
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('vader_lexicon')
   ```

## Usage

### Try the Live Demo

**[Access the deployed application](https://multimodal-review-prediction.streamlit.app/)** - No setup required!

### Running the Web Application Locally

```bash
streamlit run app.py
```

This launches an interactive web interface where you can:
- Select from sample reviews
- View extracted features (text analysis, metadata, image info)
- Get real-time helpfulness predictions
- Explore the model's decision-making process

### Training Your Own Model

1. **Data Preprocessing**
   ```bash
   python data_clean.py
   ```

2. **Feature Engineering**
   Open and run `data_preprocessing.ipynb` to extract:
   - BERT embeddings using DistilBERT
   - Image features using ResNet
   - Text quality metrics (sentiment, readability)
   - Temporal and metadata features

3. **Model Training**
   Open and run `final.ipynb` to:
   - Apply PCA dimensionality reduction
   - Train the CatBoost regressor
   - Evaluate model performance
   - Save trained models

## Features Used

### Text Features
- **BERT Embeddings**: 768-dimensional semantic representations
- **Sentiment Score**: VADER sentiment analysis
- **Readability**: Flesch reading ease score
- **Review Length**: Word count and character count
- **Punctuation Count**: Number of punctuation marks

### Image Features
- **Deep Learning Embeddings**: ResNet-based feature extraction
- **Visual Content Analysis**: Automated image quality assessment

### Metadata Features
- **Rating Information**: Review rating, average product rating, rating deviation
- **Verification**: Verified purchase status
- **Temporal**: Days since review was posted
- **Product Metrics**: Average quality score, total number of ratings
- **Category**: Product main category

## Model Performance

The system uses CatBoost (Categorical Boosting) as the final predictor, which is particularly effective for:
- Handling categorical features
- Avoiding overfitting
- Providing interpretable results
- Fast training and inference

Key evaluation metrics:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Pearson Correlation

## Technical Details

### Data Pipeline
1. **Raw Data**: Amazon Reviews 2023 dataset (All Beauty category)
2. **Preprocessing**: Text cleaning, image processing, missing value handling
3. **Feature Engineering**: Multi-modal feature extraction
4. **Normalization**: Standard scaling for numerical features
5. **Dimensionality Reduction**: PCA for high-dimensional embeddings

### Model Architecture
- **Base Model**: CatBoost Regressor
- **Input Features**: ~300+ dimensional feature vector
- **Output**: Continuous helpfulness score (0-1 range)
- **Training**: Gradient boosting with categorical feature support

### Preprocessing Steps
- Text normalization and cleaning
- BERT tokenization and embedding extraction
- Image resizing and feature extraction
- Temporal feature engineering
- Cross-validation for robust evaluation

## Results

The multimodal approach significantly outperforms text-only baselines by incorporating:
- Visual information from product/review images
- Rich semantic representations from BERT
- Comprehensive metadata analysis
- Temporal dynamics of review helpfulness

## Dependencies

Key libraries:
- **Machine Learning**: scikit-learn, catboost, numpy, pandas
- **Deep Learning**: torch, transformers, sentence-transformers
- **NLP**: nltk, textstat
- **Computer Vision**: opencv-python, scikit-image
- **Web App**: streamlit
- **Visualization**: matplotlib, seaborn, plotly

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is for educational and research purposes.

## Acknowledgments

- Amazon Reviews 2023 dataset by McAuley Lab
- Hugging Face for transformer models
- CatBoost team for the gradient boosting implementation

---

*Built as part of the Computational Data Science course at SUTD*
