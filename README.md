# Text Classification 

## 🎯 Objective
This project demonstrates **text classification and sentiment analysis** using two machine learning approaches:
1. **Logistic Regression** on the **IMDB movie reviews dataset**.
2. **Support Vector Machine (SVM)** on a **combined real-world dataset** (environmental, social, and technological topics).

The main goal is to classify text into *positive* or *negative* sentiments and visualize dataset characteristics such as class distribution and text length.

---

## 🧩 Key Features

### 🔹 Part 1 – IMDB Sentiment Classification
- Loads the **IMDB dataset** directly from Hugging Face.
- Detects dataset issues:
  - Missing values, duplicate texts, class imbalance, outliers.
- Applies **TF-IDF vectorization** on movie reviews.
- Trains a **Logistic Regression model** for binary sentiment classification.
- Evaluates performance using:
  - **Accuracy** and **Classification Report**.
- Visualizes:
  - Class distribution of reviews.
  - Review length histogram.
- Flags **potentially mislabeled reviews**.
- Provides a **custom query prediction** function for user input.

### 🔹 Part 2 – SVM Classification on Multiple Datasets
- Combines the following CSV datasets:
  - `BiodiversityConservationDataset.csv`
  - `ClimateChangeDataset.csv`
  - `SocialJusticeDataset.csv`
  - `SpaceExplorationDataset.csv`
  - `TechnologyImpactDataset.csv`
- Preprocesses the data:
  - Drops missing and duplicate sentences.
  - Filters only `positive` and `negative` sentiment labels.
- Extracts features using **TF-IDF** (with bigrams).
- Trains an **SVM classifier (linear kernel)**.
- Generates performance metrics and visualizations.
- Provides a **custom query testing function** for user sentences.

---

## ⚙️ Workflow

### **Part 1 – Logistic Regression Model**
1. Load IMDB dataset → `load_dataset("imdb")`
2. Clean and inspect data.
3. Split data → train/test sets.
4. Extract TF-IDF features → `TfidfVectorizer(max_features=20000, ngram_range=(1,2))`
5. Train **Logistic Regression** model → `LogisticRegression(max_iter=1000)`
6. Evaluate → Accuracy & Classification Report.
7. Visualize → Class balance and review lengths.
8. Predict → Sentiment for user-provided queries.

### **Part 2 – SVM Model**
1. Load and merge 5 domain-specific CSV files.
2. Clean dataset (remove missing/duplicate rows).
3. Vectorize sentences using **TF-IDF**.
4. Train **SVM classifier** → `SVC(kernel='linear', probability=True)`
5. Evaluate → Accuracy & Classification Report.
6. Visualize → Sentiment distribution & sentence length histogram.
7. Predict → Sentiment and confidence score for user queries.

---

## 🛠️ Libraries and Dependencies

| Library | Description |
|----------|--------------|
| **pandas** | For loading and preprocessing CSV data |
| **numpy** | For numerical operations |
| **matplotlib** | For data visualization |
| **datasets (Hugging Face)** | For loading IMDB dataset |
| **scikit-learn** | For ML models, TF-IDF, and evaluation metrics |
| **LogisticRegression** | For IMDB text sentiment classification |
| **SVC (SVM)** | For multi-dataset text classification |
| **TfidfVectorizer** | For converting text into numeric features |
| **train_test_split** | For dataset partitioning |
| **classification_report, accuracy_score** | For performance evaluation |
| **Google Colab Drive API** | To mount and read datasets from Google Drive |

---

## 🧪 Example Output

### ✅ Logistic Regression (IMDB)
