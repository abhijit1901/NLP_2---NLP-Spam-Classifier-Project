# SMS Spam Classification with Scikit-learn & Gensim

This repository provides a hands-on comparison of different Natural Language Processing (NLP) techniques for classifying SMS messages as **spam** or **ham** (not spam). Each Jupyter notebook implements a complete pipeline, from text preprocessing to model training and evaluation, showcasing a different approach to feature extraction.

The project explores three fundamental methods:

  * **Bag of Words (BoW)** with Stemming
  * **TF-IDF** with Lemmatization
  * **Average Word2Vec Embeddings** with Lemmatization

-----

## Table of Contents

  - [Core Concepts Demonstrated](https://www.google.com/search?q=%23-core-concepts-demonstrated)
  - [Getting Started](https://www.google.com/search?q=%23-getting-started)
  - [Notebook Descriptions](https://www.google.com/search?q=%23-notebook-descriptions)
  - [Results Summary](https://www.google.com/search?q=%23-results-summary)

-----

## Core Concepts Demonstrated

  * **Text Preprocessing**:
      * Stopword Removal
      * Stemming (`nltk.PorterStemmer`)
      * Lemmatization (`nltk.WordNetLemmatizer`)
  * **Feature Extraction**:
      * Bag of Words (`sklearn.feature_extraction.text.CountVectorizer`)
      * TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`)
      * Word2Vec Embeddings (`gensim.models.Word2Vec`)
  * **Machine Learning Models**:
      * Multinomial Naive Bayes (`sklearn.naive_bayes.MultinomialNB`)
      * Random Forest (`sklearn.ensemble.RandomForestClassifier`)
  * **Model Evaluation**:
      * Accuracy Score
      * Classification Report (Precision, Recall, F1-Score)

-----

## Getting Started

Follow these instructions to set up the environment and run the notebooks.

### Prerequisites

  * Python 3.x
  * Jupyter Notebook or JupyterLab

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```
2.  **Install the required libraries:**
    ```bash
    pip install pandas nltk scikit-learn gensim tqdm
    ```
3.  **Download NLTK Data:**
    Run the following in a Python interpreter or Jupyter cell to download the necessary NLTK models.
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```

### Dataset

Ensure the `SMSSpamCollection` dataset file is available in the root directory of the project.

-----

## Notebook Descriptions

### 1\. Bag of Words + Stemming

**File:** `1_spam_and_ham_project_using_BOW_and_stemming.ipynb`

This notebook builds a spam classifier using the classic **Bag of Words (BoW)** model. It represents each message based on the frequency of words it contains.

#### Workflow:

1.  **Preprocessing**: Text is cleaned by removing non-alphabetic characters, converting to lowercase, and reducing words to their root form using **stemming**.
2.  **Feature Extraction**: `CountVectorizer` from Scikit-learn is used to convert the cleaned text into a matrix of token counts. Both unigrams and bigrams are considered (`ngram_range=(1,2)`).
3.  **Model Training**: A **Multinomial Naive Bayes** classifier is trained on the BoW feature matrix.
4.  **Evaluation**: The model's performance is measured using accuracy and a detailed classification report.

### 2\. TF-IDF + Lemmatization

**File:** `2_spam_and_ham_using_TF-IDF_and_lemmatization.ipynb`

This notebook improves upon the BoW model by using **Term Frequency-Inverse Document Frequency (TF-IDF)**, which weights words based on their importance in the corpus, not just their frequency.

#### Workflow:

1.  **Preprocessing**: Text is cleaned, and words are normalized to their meaningful base form using **lemmatization**, which is generally more accurate than stemming.
2.  **Feature Extraction**: `TfidfVectorizer` from Scikit-learn converts the text into a matrix where each cell represents the TF-IDF score of a word.
3.  **Model Training**: A **Multinomial Naive Bayes** classifier is trained on the resulting TF-IDF features.
4.  **Evaluation**: The model's performance is evaluated and compared to the BoW approach.

### 3\. Average Word2Vec Embeddings

**File:** `3_spam_and_ham_using_word2vec_and_avgWord2Vec.ipynb`

This notebook moves from frequency-based models to semantic embeddings using **Word2Vec**. Instead of counting words, this approach captures the contextual meaning of words in a dense vector space.

#### Workflow:

1.  **Preprocessing**: Text is cleaned and lemmatized.
2.  **Word Embedding Model**: A **Word2Vec** model is trained from scratch on the SMS corpus using the `gensim` library.
3.  **Feature Extraction**: For each SMS message, a single feature vector is created by **averaging the Word2Vec vectors** of all the words present in it.
4.  **Model Training**: A **Random Forest Classifier** is trained on the averaged embedding vectors.
5.  **Evaluation**: The model's performance is analyzed to see how a meaning-based approach compares to frequency-based methods.

-----

## Results Summary

The performance of each model was evaluated on a held-out test set (20% of the data). Below is a summary of their accuracy scores.

| \# | Model | Preprocessing | Feature Extraction | Classifier | Accuracy |
|---|---|---|---|---|---|
| 1 | Bag of Words | Stemming | CountVectorizer | Naive Bayes | **98.3%** |
| 2 | TF-IDF | Lemmatization | TfidfVectorizer | Naive Bayes | **97.7%** |
| 3 | Word2Vec | Lemmatization | Average Embeddings | Random Forest | **96.9%** |

While the simpler BoW and TF-IDF models performed slightly better in terms of raw accuracy, the Word2Vec approach demonstrates how to use semantic embeddings, which can be more powerful for complex NLP tasks where context and meaning are crucial.
