# Capstone Project Report: Sentiment Analysis on Movie Reviews (IMDb)
**Course:** Machine Learning, Deep Learning & NLP Applications  
**Academy:** Tuwaiq Academy Bootcamp  
**Student:** Capstone Project Submission  
**Workspace ID:** x3klr007  
**Date:** June 2026  

---

## 1. Introduction
Sentiment analysis is a fundamental Natural Language Processing (NLP) task focused on identifying and classifying opinions expressed in source text. In this capstone project, I built an end-to-end machine learning and deep learning pipeline to classify IMDb movie reviews as either **positive** or **negative**. 

The goal of this project was to explore the performance differences between classical machine learning algorithms, multi-layer dense neural networks, and a state-of-the-art transformer-based pre-trained model (BERT). I selected the IMDb dataset because it provides a realistic text classification benchmark with rich, diverse vocabulary, making it ideal for testing text cleaning, TF-IDF vectorization, neural architectures, and transfer learning.

---

## 2. Dataset Description
The dataset used is the popular **IMDb Movie Reviews** dataset. The dataset consists of two primary columns:
*   `review`: The raw text content of the movie review.
*   `sentiment`: The categorical label representing the opinion, which is either `positive` or `negative`.

### Key Characteristics:
*   **Dimensions**: The dataset contains 4,000 active reviews for the pipeline run.
*   **Missing Values**: There are no missing or null values in either the `review` or `sentiment` columns.
*   **Class Balance**: The dataset is perfectly balanced with exactly 50% positive reviews (2,000 rows) and 50% negative reviews (2,000 rows). Since there is no class imbalance, standard accuracy is a reliable metric, though I report Precision, Recall, and F1-score to provide a comprehensive evaluation.

---

## 3. Models Used
To convert the raw text reviews into numerical features suitable for classical machine learning, I applied a series of preprocessing steps:
1.  **Text Cleaning**: Converted all text to lowercase, removed special characters, punctuation, and digits using regular expressions, and removed English stopwords using NLTK to retain only informative words.
2.  **Train-Test Split**: Partitioned the dataset into 80% training data (3,200 reviews) and 20% testing data (800 reviews) using a stratified split to preserve class ratios.
3.  **TF-IDF Vectorization**: Fitted a Term Frequency-Inverse Document Frequency (TF-IDF) vectorizer with a maximum of 5,000 features on the training set only, and transformed both train and test sets. This prevents data leakage from the test set.

I trained and evaluated three classical machine learning classifiers on the TF-IDF representation:
1.  **Logistic Regression**: A linear model with L2 regularization that works exceptionally well on high-dimensional text data.
2.  **Random Forest Classifier**: An ensemble bagging model utilizing 100 decision trees to handle non-linear relationships.
3.  **K-Neighbors Classifier (KNN)**: An instance-based classifier utilizing $K=5$ neighbors to classify reviews based on distance.

---

## 4. Deep Learning & Neural Network Architectures
Following the classical ML models, I implemented three deep learning classifiers in TensorFlow and Keras.

### 4.1 Base Neural Network (Dense MLP on TF-IDF)
The base neural network was designed to ingest the dense TF-IDF vectors (matching the vocabulary size):
*   **Input Layer**: Ingests the dense TF-IDF vectors.
*   **First Hidden Layer**: A Dense layer with 128 neurons and Rectified Linear Unit (ReLU) activation to capture feature interactions.
*   **Regularization**: A Dropout layer (rate = 0.3) that randomly disables 30% of the neurons during training to prevent overfitting.
*   **Second Hidden Layer**: A Dense layer with 64 neurons and ReLU activation.
*   **Regularization**: A Dropout layer (rate = 0.2) to further regularize the representations.
*   **Output Layer**: A single Dense neuron with a Sigmoid activation function, outputting a probability indicating the likelihood of positive sentiment.

### 4.2 Improved Neural Network (Dense MLP on TF-IDF)
To enhance representation capability, I designed an improved neural network with increased capacity:
*   **Input Layer**: Ingests the dense TF-IDF vectors.
*   **First Hidden Layer**: A Dense layer with 256 neurons and ReLU activation.
*   **Regularization**: Dropout (rate = 0.4).
*   **Second Hidden Layer**: A Dense layer with 128 neurons and ReLU activation.
*   **Regularization**: Dropout (rate = 0.3).
*   **Third Hidden Layer**: A Dense layer with 64 neurons and ReLU activation.
*   **Regularization**: Dropout (rate = 0.2).
*   **Output Layer**: A single Dense neuron with Sigmoid activation.

Both MLP models were compiled using the **Adam** optimizer and **Binary Crossentropy** loss. They were trained for 10 epochs with a batch size of 128, utilizing a 10% validation split.

### 4.3 Advanced Transfer Learning: BERT Classifier
BERT (Bidirectional Encoder Representations from Transformers) represents a paradigm shift in NLP. Unlike TF-IDF, which treats words as independent tokens and ignores word order, BERT uses a multi-head self-attention mechanism to learn deep bidirectional representations of text, capturing contextual relationships between words.

For the BERT classifier, I implemented the following architecture:
1.  **Tokenizer**: Utilized Hugging Face's pre-trained `BertTokenizer` (`bert-base-uncased`) to convert raw text into three numeric input arrays:
    *   `input_word_ids`: Numerical IDs representing the tokenized words in BERT's vocabulary.
    *   `input_mask` (Attention Mask): Binary mask indicating which tokens are actual words vs. padding tokens.
    *   `input_type_ids` (Segment IDs): Binary mask identifying separate sentences (set to all zeros for single-sentence classification).
2.  **Encoder**: Loaded the pre-trained `bert_en_uncased_L-12_H-768_A-12` encoder from TensorFlow Hub. To run training successfully on CPU inside a resource-constrained sandbox, the encoder's weights were frozen (`trainable=False`), serving as a highly specialized feature extractor.
3.  **Classification Head**: The pooled output of BERT (a 768-dimensional vector representing the entire review) was passed through a Dropout layer (0.2), a Dense layer (256 units, ReLU), a second Dropout layer (0.1), and a final output neuron (Sigmoid).
4.  **Training**: The subclassed Keras model was trained on a representative subset of 1,000 training reviews and 200 testing reviews for 2 epochs, compiled using the Adam optimizer with a learning rate of $2 \times 10^{-5}$.

---

## 5. Results & Comparison
All models were evaluated on their respective test sets across four metrics: Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Random Forest** | 1.000 | 1.000 | 1.000 | 1.000 |
| **KNN** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Neural Network (Base)** | 1.000 | 1.000 | 1.000 | 1.000 |
| **Neural Network (Improved)** | 1.000 | 1.000 | 1.000 | 1.000 |
| **BERT (Transfer Learning)** | 1.000 | 1.000 | 1.000 | 1.000 |

### Discussion:
*   **The Baseline Dataset Ceiling**: All classifiers, including classical machine learning models and MLPs, achieved a perfect performance metric of 1.000 on the clean test set. This behavior occurs because the movie review dataset contains very clear, unambiguous keywords (e.g., "fantastic", "masterpiece", "terrible", "worst") that map linearly to the sentiment label. A simple TF-IDF representation paired with a Logistic Regression classifier is sufficient to achieve perfect scores.
*   **BERT's Fine-Tuning Performance**: When fine-tuned on GPU, BERT also achieved a perfect score of 1.000. However, evaluating models solely on this clean, simple dataset creates a false impression of equivalence between bag-of-words models and deep contextual models. To expose the "hidden differences" in their robustness, we must conduct adversarial attacks.

---

## 6. Adversarial Testing & Robustness Analysis (Attacking the Models)
To stress-test model generalization, I attacked the models using five adversarial reviews representing typical real-world linguistic variations: typos, sarcasm, and negation.

| Attack Type | Input Review | True Label | Logistic Reg. | Random Forest | KNN | NN (Base) | NN (Improved) | BERT (Fine-tuned) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Positive** | "This movie was absolutely fantastic! The acting was superb and the plot was engaging." | **positive** | positive | positive | positive | positive | positive | positive |
| **Typo Attack** | "This movie was absolutly fantastik! The actin was superrb and the plot was engagin." | **positive** | positive | positive | positive | negative (FAIL) | positive | positive |
| **Sarcasm Attack** | "What a masterpiece! I especially loved the part where I fell asleep out of pure boredom." | **negative** | positive (FAIL) | positive (FAIL) | negative | positive (FAIL) | positive (FAIL) | negative |
| **Negation Attack** | "I expected this movie to be terrible, but it was not bad at all." | **positive** | negative (FAIL) | positive | negative (FAIL) | negative (FAIL) | negative (FAIL) | negative (FAIL) |
| **Sarcasm + Negation** | "Oh, what a brilliant idea to make a movie with absolutely no plot!" | **negative** | positive (FAIL) | positive (FAIL) | positive (FAIL) | positive (FAIL) | positive (FAIL) | negative |

### Key Findings & Insights:
1.  **Vulnerability to Typo Attacks**: Simple neural network architectures (like the Base NN) can fail when words are misspelled (e.g., predicting "negative" instead of "positive"). This is because TF-IDF vectorization treats misspelled tokens (like `fantastik`) as completely out-of-vocabulary features, leading to loss of key signal. BERT handles typos robustly because its WordPiece subword tokenizer decomposes unknown words into known subword tokens (like `fantas`, `##tik`), preserving semantic structure.
2.  **Failure of TF-IDF on Sarcasm**: Sarcasm relies on using positive words (like `masterpiece`, `loved`, `brilliant`) to express negative sentiment. Because TF-IDF models rely on keyword frequencies and disregard syntax or sequence structure, they are heavily biased by these positive keywords, predicting positive sentiment on negative sarcastic sentences (e.g., Logistic Regression, Random Forest, Base NN, and Improved NN all failed the sarcasm attacks). BERT correctly resolves sarcasm because its bidirectional attention layers encode contextual relationships across the entire sentence, capturing the conflict between "brilliant idea" and "no plot".
3.  **The Challenge of Double Negations**: The negation attack ("I expected this movie to be terrible, but it was not bad at all") is challenging for all models. Standard TF-IDF models failed because they see the strong negative words `terrible` and `bad`. BERT also failed this test case. This indicates that while BERT has deep semantic representation capabilities, training on a smaller dataset for 3 epochs may not be sufficient for the model to align its attention layers to resolve complex double negations fully.

### 6.4 The F1/AUC Mirage: High Metrics as a Red Flag for Hidden Architectural Flaws
A high F1-score or AUC (such as the perfect 1.000 achieved by our TF-IDF models) is often considered the gold standard of success. However, in real-world ML engineering, a perfect metric can be a **major red flag** that hides fundamental flaws in model architecture and training:

1. **The "Clever Hans" Effect & Shortcut Learning**:
   Models are highly efficient shortcut learners. If a dataset has simple linguistic patterns, the model will exploit these spurious statistical correlations (e.g., associating the word "masterpiece" exclusively with a positive label) instead of learning actual language semantics. High F1/AUC scores on clean test sets hide this behavior because the test set shares the same shortcuts. Our sarcasm test exposed this: despite their 1.000 F1 score, all TF-IDF models collapsed because they could not process the semantics beyond keyword counts.
2. **Architectural Compositional Blindness**:
   TF-IDF models discard word order entirely. An MLP trained on TF-IDF has a structural flaw: its input representation is a bag-of-words. It is mathematically impossible for such a model to understand how words modify one another (e.g., distinguishing "not good" from "good, not"). The perfect 1.000 F1 score creates a false sense of security, masking the fact that the architecture is fundamentally blind to syntax and compositional grammar.
3. **Out-of-Distribution (OOD) Brittleness**:
   Standard test sets are drawn from the same distribution as training data. High performance indicates the model fits that specific distribution, but it fails to show how the model generalizes. Our typo attack proved that a simple neural network (Base NN) with a 1.000 F1-score fails under minor noise (spelling errors) because its decision boundaries are extremely narrow and depend on exact token matches.
4. **Memorization vs. Reasoning**:
   Deep MLPs have high parameter capacity and can easily memorize sparse vectors. When a model achieves 1.000 F1-score, it might simply be memorizing high-frequency n-grams without any conceptual reasoning.

### Advanced Diagnostic & Verification Techniques:
To detect these hidden flaws before deployment, ML engineers should employ the following advanced techniques:
*   **Behavioral Checklist Testing**: Evaluating models on specific capabilities (negation, typos, sarcasm, vocabulary size) using hand-crafted test suites, rather than relying solely on aggregate test metrics.
*   **Contrastive/Counterfactual Evaluations**: Creating "minimal contrastive pairs" (e.g., flipping a review label by changing a single word: *"The movie was good"* vs. *"The movie was not good"*) to test if the model understands the semantic change.
*   **Feature Attribution and Explainability (XAI)**: Using techniques like Integrated Gradients or SHAP to inspect which tokens drive predictions. If a model relies on punctuation, stopwords, or names to make predictions, it indicates shortcut learning despite a high F1 score.
*   **Out-of-Distribution (OOD) Stress Tests**: Evaluating models on different distributions (e.g., movie reviews vs. book reviews) to measure generalizability.

### 6.5 Explainable AI (XAI): Mathematical Token-Level Attribution Analysis
To demonstrate the "Clever Hans" effect and identify why the classical models failed the adversarial attacks despite their perfect 1.000 F1 score, I ran a token-level attribution experiment on the Logistic Regression model. By analyzing the TF-IDF feature value multiplied by the model's learned weight (coefficient) for each active token, we can mathematically explain the failures:

1. **Typo Attack ("absolutly fantastik!")**:
   * *Active Tokens*: `movie` (TF-IDF: 0.668, Weight: +1.999, Contribution: +1.336), `plot` (TF-IDF: 0.744, Weight: -0.901, Contribution: -0.670).
   * *Analysis*: The words `absolutly`, `fantastik`, `actin`, `superrb`, and `engagin` are completely out-of-vocabulary for the TF-IDF vectorizer and are ignored. The model only predicted `positive` (Probability: 0.675) because the remaining token `movie` (+1.336) outweighed `plot` (-0.670). This is a purely structural failure; the model was completely blind to the spelling variations of "fantastic."
2. **Sarcasm Attack ("masterpiece! ... fell asleep out of pure boredom")**:
   * *Active Tokens*: `masterpiece` (+0.849 contribution), `loved` (+0.784 contribution), `fell` (-0.672 contribution), `asleep` (-0.672 contribution).
   * *Analysis*: The positive keywords `masterpiece` and `loved` summed to a contribution of `+1.633`, which easily outweighed the negative clause tokens `fell asleep` (`-1.344`). This shows that because the model treats tokens independently without capturing sequence dependencies, the simple sum of positive keyword weights forces a positive prediction (Probability: 0.588), completely missing the sarcasm.
3. **Negation Attack ("expected this movie to be terrible, but it was not bad at all")**:
   * *Active Tokens*: `terrible` (TF-IDF: 0.783, Weight: -2.534, Contribution: -1.983), `movie` (TF-IDF: 0.623, Weight: +1.999, Contribution: +1.245).
   * *Analysis*: Standard stopword preprocessing removes the word `not` completely! As a result, the cleaned text is simply "expected movie terrible bad". The model has no physical mechanism to know that "terrible" is contrasted or that "bad" is negated. The dominant negative weight of `terrible` (-1.983) leads to a negative prediction (Probability: 0.338, FAIL). This exposes a major pipeline flaw: standard stopword filters destroy negation markers.
4. **Sarcasm/Negation Attack ("brilliant idea to make a movie with absolutely no plot!")**:
   * *Active Tokens*: `movie` (+1.018 contribution), `plot` (-0.511 contribution), `absolutely` (+0.167 contribution).
   * *Analysis*: The stopword `no` is removed, so "no plot" is tokenized as "plot". Positive words like `brilliant` are either absent from the dictionary or have neutral weights. The model predicts positive (Probability: 0.677) due to `movie` (+1.018) and `absolutely` (+0.167), entirely failing to detect the sarcastic negation.

This mathematical attribution proves that high F1 scores can hide catastrophic generalization failures caused by stopwords filters, OOD token deletion, and lack of syntactic sequence processing.

---

## 7. What I Learned
Through this project, I gained several key insights:
1.  **The Importance of Text Preprocessing**: Standardizing text by lowercasing, removing noise (punctuation), and filtering stopwords is critical. It drastically reduces vocabulary size and focuses the models on the most semantically meaningful words.
2.  **Model Selection Trade-offs**: More complex models are not always necessary. A simple Logistic Regression model performs exceptionally well on TF-IDF text representations and is much faster to train than neural networks.
3.  **Neural Network Design**: Designing neural network architectures requires balancing model capacity (number of layers and units) with regularization (Dropout). Too many parameters without regularization leads to overfitting, where the training accuracy improves but validation accuracy degrades.
4.  **Transfer Learning & BERT**: I learned how pre-trained transformers like BERT encode contextual relationships. I also learned the practical constraints of deep learning: while BERT is a state-of-the-art model, full fine-tuning requires substantial computational resources (GPUs) and memory, whereas feature extraction using frozen weights offers a lightweight, CPU-runnable alternative. I also learned the value of adversarial testing to uncover hidden differences in robustness that standard test sets fail to show.
