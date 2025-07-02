# spam_detector.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# Step 1: Load and clean data
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 2: Convert labels to binary
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Create train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Step 4: Create a pipeline (TF-IDF + Naive Bayes)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Step 5: Train the model
model.fit(X_train, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test)
print("Evaluation Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the model for website use
joblib.dump(model, "spam_model.pkl")
print("✅ Model saved as 'spam_model.pkl'")
