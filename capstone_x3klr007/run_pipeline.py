import torch
if torch.cuda.is_available():
    try:
        torch.cuda.init()
    except Exception:
        pass

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout

# Setup directories
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

# Download NLTK stopwords
try:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords', quiet=True)
    stop_words = set(stopwords.words('english'))
except Exception as e:
    print(f"Warning: NLTK download failed ({e}). Using fallback stopwords list.")
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

def run_ml_and_nn_pipeline():
    print("--- Starting Full Pipeline ---")
    
    # 1. Load and Explore
    if not os.path.exists('data/imdb_reviews.csv'):
        raise FileNotFoundError("Dataset data/imdb_reviews.csv not found! Run prepare_data.py first.")
        
    df = pd.read_csv('data/imdb_reviews.csv')
    print(f"Dataset shape: {df.shape}")
    print("Checking class balance:")
    print(df['sentiment'].value_counts())
    
    print("Cleaning text...")
    df['clean_text'] = df['review'].apply(clean_text)
    
    X = df['clean_text']
    y = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Vectorizing text...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_raw)
    X_test_tfidf = tfidf.transform(X_test_raw)
    
    # Save TF-IDF for later
    joblib.dump(tfidf, 'tfidf_vectorizer.joblib')
    
    # Convert to dense for Neural Network
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()
    
    # Step 1: Classical ML Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    summary_results = []
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        summary_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        })
        
        # Plot Confusion Matrix
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'plots/cm_{name.replace(" ", "_").lower()}.png')
        plt.close()
        
    # Step 2: Neural Network (Base)
    print("Training Base Neural Network...")
    nn_base = Sequential([
        Input(shape=(X_train_dense.shape[1],)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_base.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_base = nn_base.fit(X_train_dense, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=1)
    
    y_pred_nn = (nn_base.predict(X_test_dense) > 0.5).astype(int)
    acc_base = accuracy_score(y_test, y_pred_nn)
    prec_base = precision_score(y_test, y_pred_nn)
    rec_base = recall_score(y_test, y_pred_nn)
    f1_base = f1_score(y_test, y_pred_nn)
    
    summary_results.append({
        "Model": "Neural Network (Base)",
        "Accuracy": acc_base,
        "Precision": prec_base,
        "Recall": rec_base,
        "F1-score": f1_base
    })
    
    # Save confusion matrix for Base NN
    plt.figure(figsize=(5, 4))
    cm_base = confusion_matrix(y_test, y_pred_nn)
    sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Neural Network (Base)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/cm_neural_network_(base).png')
    plt.close()
    
    # Step 2 Improvement: More layers and different dropout
    print("Training Improved Neural Network...")
    nn_imp = Sequential([
        Input(shape=(X_train_dense.shape[1],)),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    nn_imp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history_imp = nn_imp.fit(X_train_dense, y_train, epochs=10, batch_size=128, validation_split=0.1, verbose=1)
    
    y_pred_nn_imp = (nn_imp.predict(X_test_dense) > 0.5).astype(int)
    acc_imp = accuracy_score(y_test, y_pred_nn_imp)
    prec_imp = precision_score(y_test, y_pred_nn_imp)
    rec_imp = recall_score(y_test, y_pred_nn_imp)
    f1_imp = f1_score(y_test, y_pred_nn_imp)
    
    summary_results.append({
        "Model": "Neural Network (Improved)",
        "Accuracy": acc_imp,
        "Precision": prec_imp,
        "Recall": rec_imp,
        "F1-score": f1_imp
    })
    
    # Save confusion matrix for Improved NN
    plt.figure(figsize=(5, 4))
    cm_imp = confusion_matrix(y_test, y_pred_nn_imp)
    sns.heatmap(cm_imp, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Neural Network (Improved)')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('plots/cm_neural_network_(improved).png')
    plt.close()
    
    # Save curves
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_base.history['accuracy'], label='Base Train')
    plt.plot(history_base.history['val_accuracy'], label='Base Val')
    plt.plot(history_imp.history['accuracy'], label='Imp Train')
    plt.plot(history_imp.history['val_accuracy'], label='Imp Val')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history_base.history['loss'], label='Base Train')
    plt.plot(history_base.history['val_loss'], label='Base Val')
    plt.plot(history_imp.history['loss'], label='Imp Train')
    plt.plot(history_imp.history['val_loss'], label='Imp Val')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/nn_curves.png')
    plt.close()
    
    # Try running the BERT model (Transfer Learning) on a small subset
    print("Checking if BERT training is possible...")
    bert_results = run_bert_subset(X_train_raw, X_test_raw, y_train, y_test, models, nn_base, nn_imp, tfidf, clean_text)
    if bert_results is not None:
        summary_results.append(bert_results)
        
    # Final Summary
    summary_df = pd.DataFrame(summary_results)
    print("\nFinal Results Summary:")
    print(summary_df)
    summary_df.to_csv('results/final_summary.csv', index=False)
    
def run_bert_subset(X_train_raw, X_test_raw, y_train, y_test, models, nn_base, nn_imp, tfidf, clean_text):
    try:
        print("Importing PyTorch and Hugging Face Transformers...")
        import torch
        from transformers import BertTokenizer, BertForSequenceClassification
        from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
        from torch.optim import AdamW
        import time
        
        print("Using GPU:", torch.cuda.get_device_name(0))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("Loading pre-trained BERT elements...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)
        
        def tokenize_texts(texts):
            return tokenizer(list(texts), max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            
        print("Tokenizing reviews (2000 train, 500 test)...")
        train_tokens = tokenize_texts(X_train_raw.iloc[:2000])
        test_tokens = tokenize_texts(X_test_raw.iloc[:500])
        
        y_train_subset = torch.tensor(y_train.iloc[:2000].values, dtype=torch.long)
        y_test_subset = torch.tensor(y_test.iloc[:500].values, dtype=torch.long)
        
        # Datasets & Dataloaders
        train_dataset = TensorDataset(train_tokens['input_ids'], train_tokens['attention_mask'], train_tokens['token_type_ids'], y_train_subset)
        test_dataset = TensorDataset(test_tokens['input_ids'], test_tokens['attention_mask'], test_tokens['token_type_ids'], y_test_subset)
        
        train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
        test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=16)
        
        optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        
        epochs = 3
        history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': []
        }
        
        print("Training BERT model on GPU (Full Fine-Tuning)...")
        for epoch in range(epochs):
            start_time = time.time()
            model.train()
            total_train_loss = 0
            correct_train = 0
            total_train = 0
            
            for step, batch in enumerate(train_dataloader):
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = [b.to(device) for b in batch]
                
                optimizer.zero_grad()
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits
                
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                correct_train += (preds == b_labels).sum().item()
                total_train += b_labels.size(0)
                
            train_loss = total_train_loss / len(train_dataloader)
            train_acc = correct_train / total_train
            
            # Validation
            model.eval()
            total_val_loss = 0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for batch in test_dataloader:
                    b_input_ids, b_input_mask, b_token_type_ids, b_labels = [b.to(device) for b in batch]
                    outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs.loss
                    logits = outputs.logits
                    
                    total_val_loss += loss.item()
                    
                    preds = torch.argmax(logits, dim=1)
                    correct_val += (preds == b_labels).sum().item()
                    total_val += b_labels.size(0)
                    
            val_loss = total_val_loss / len(test_dataloader)
            val_acc = correct_val / total_val
            
            history['loss'].append(train_loss)
            history['accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {elapsed:.1f}s - loss: {train_loss:.4f} - accuracy: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_accuracy: {val_acc:.4f}")
            
        print("Evaluating BERT model...")
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_dataloader:
                b_input_ids, b_input_mask, b_token_type_ids, b_labels = [b.to(device) for b in batch]
                outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(b_labels.cpu().numpy())
                
        acc = accuracy_score(true_labels, predictions)
        prec = precision_score(true_labels, predictions, zero_division=0)
        rec = recall_score(true_labels, predictions, zero_division=0)
        f1 = f1_score(true_labels, predictions, zero_division=0)
        
        # Save BERT confusion matrix
        plt.figure(figsize=(5, 4))
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix - BERT (Transfer Learning)')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('plots/cm_bert_(transfer_learning).png')
        plt.close()
        
        # Save BERT learning curves
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history["accuracy"], label="Train Acc")
        plt.plot(history["val_accuracy"], label="Val Acc")
        plt.title("BERT Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history["loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Val Loss")
        plt.title("BERT Loss")
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig("plots/bert_curves.png")
        plt.close()
        
        bert_results = {
            "Model": "BERT (Transfer Learning)",
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-score": f1
        }
        pd.DataFrame([bert_results]).to_csv("results/optimized_bert_results.csv", index=False)
        
        # --- PERFORM ADVERSARIAL ATTACKS (ROBUSTNESS TESTS) ---
        print("\n--- Performing Adversarial Attacks (Model Robustness Tests) ---")
        adversarial_cases = [
            {"text": "This movie was absolutely fantastic! The acting was superb and the plot was engaging.", "label": "positive", "type": "Standard Positive"},
            {"text": "This movie was absolutly fantastik! The actin was superrb and the plot was engagin.", "label": "positive", "type": "Typo Attack"},
            {"text": "What a masterpiece! I especially loved the part where I fell asleep out of pure boredom.", "label": "negative", "type": "Sarcasm Attack"},
            {"text": "I expected this movie to be terrible, but it was not bad at all.", "label": "positive", "type": "Negation Attack"},
            {"text": "Oh, what a brilliant idea to make a movie with absolutely no plot!", "label": "negative", "type": "Sarcasm/Negation Attack"}
        ]
        
        adv_texts = [case["text"] for case in adversarial_cases]
        cleaned_adv_texts = [clean_text(text) for text in adv_texts]
        tfidf_adv = tfidf.transform(cleaned_adv_texts).toarray()
        
        adv_results = []
        for case_idx, case in enumerate(adversarial_cases):
            row = {
                "Attack Type": case["type"],
                "Input Review": case["text"],
                "True Label": case["label"]
            }
            
            # Classical models
            for name, model_obj in models.items():
                pred = model_obj.predict(tfidf_adv[case_idx].reshape(1, -1))[0]
                row[name] = "positive" if pred == 1 else "negative"
                
            # Base NN
            pred_base_prob = nn_base.predict(tfidf_adv[case_idx].reshape(1, -1), verbose=0)[0][0]
            row["Neural Network (Base)"] = "positive" if pred_base_prob > 0.5 else "negative"
            
            # Improved NN
            pred_imp_prob = nn_imp.predict(tfidf_adv[case_idx].reshape(1, -1), verbose=0)[0][0]
            row["Neural Network (Improved)"] = "positive" if pred_imp_prob > 0.5 else "negative"
            
            # BERT (PyTorch)
            tokens_case = tokenizer(case["text"], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
            bert_in_ids = tokens_case['input_ids'].to(device)
            bert_mask = tokens_case['attention_mask'].to(device)
            bert_type_ids = tokens_case['token_type_ids'].to(device)
            
            model.eval()
            with torch.no_grad():
                out = model(bert_in_ids, token_type_ids=bert_type_ids, attention_mask=bert_mask)
                pred_bert_val = torch.argmax(out.logits, dim=1).cpu().item()
            row["BERT (Transfer Learning)"] = "positive" if pred_bert_val == 1 else "negative"
            
            adv_results.append(row)
            
        adv_df = pd.DataFrame(adv_results)
        print("\nAdversarial Attacks Evaluation Results:")
        print(adv_df.to_string(index=False))
        adv_df.to_csv("results/adversarial_summary.csv", index=False)
        
        return bert_results
        
    except Exception as e:
        print(f"BERT training skipped: error during initialization/training ({e}).")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_ml_and_nn_pipeline()
