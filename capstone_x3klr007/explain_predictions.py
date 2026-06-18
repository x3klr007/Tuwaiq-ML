import pandas as pd
import numpy as np
import re
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load NLTK stopwords
import nltk
from nltk.corpus import stopwords
try:
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception:
    stop_words = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", 
                  "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", 
                  "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", 
                  "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", 
                  "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", 
                  "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", 
                  "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", 
                  "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", 
                  "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", 
                  "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", 
                  "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# Load data and retrain Logistic Regression
print("Loading data and retraining Logistic Regression...")
df = pd.read_csv("data/imdb_reviews.csv")
df['clean_text'] = df['review'].apply(clean_text)
X = df['clean_text']
y = df['sentiment'].map({'positive': 1, 'negative': 0})

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

tfidf = joblib.load('tfidf_vectorizer.joblib')
X_train_tfidf = tfidf.transform(X_train_raw)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)

# Adversarial cases
adversarial_cases = [
    {"text": "This movie was absolutely fantastic! The acting was superb and the plot was engaging.", "type": "Standard Positive"},
    {"text": "This movie was absolutly fantastik! The actin was superrb and the plot was engagin.", "type": "Typo Attack"},
    {"text": "What a masterpiece! I especially loved the part where I fell asleep out of pure boredom.", "type": "Sarcasm Attack"},
    {"text": "I expected this movie to be terrible, but it was not bad at all.", "type": "Negation Attack"},
    {"text": "Oh, what a brilliant idea to make a movie with absolutely no plot!", "type": "Sarcasm/Negation Attack"}
]

# Explanation logic
feature_names = np.array(tfidf.get_feature_names_out())
coefficients = lr.coef_[0]

print("\n=== Explainable AI: Logistic Regression Feature Contributions ===")
for case in adversarial_cases:
    print(f"\n-------------------------------------------------------------")
    print(f"Attack Type: {case['type']}")
    print(f"Original Text: '{case['text']}'")
    cleaned = clean_text(case['text'])
    print(f"Cleaned Tokens: '{cleaned}'")
    
    # Get TF-IDF vector
    vec = tfidf.transform([cleaned]).toarray()[0]
    active_indices = np.where(vec > 0)[0]
    
    words_info = []
    total_contribution = lr.intercept_[0]
    for idx in active_indices:
        word = feature_names[idx]
        tfidf_val = vec[idx]
        coef = coefficients[idx]
        contribution = tfidf_val * coef
        total_contribution += contribution
        words_info.append({
            "Token": word,
            "TF-IDF": tfidf_val,
            "Weight (Coef)": coef,
            "Contribution": contribution
        })
        
    df_explain = pd.DataFrame(words_info)
    if not df_explain.empty:
        df_explain = df_explain.sort_values(by="Contribution", key=abs, ascending=False)
        print(df_explain.to_string(index=False))
    else:
        print("No active vocabulary words found in this review (all out-of-vocabulary).")
        
    prob = 1 / (1 + np.exp(-total_contribution))
    pred_label = "positive" if prob > 0.5 else "negative"
    print(f"\nModel Intercept: {lr.intercept_[0]:.4f}")
    print(f"Summed Log-Odds: {total_contribution:.4f} -> Predicted Probability: {prob:.4f} ({pred_label})")
