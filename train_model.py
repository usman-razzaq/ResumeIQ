import pandas as pd
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading data...")
# Load data
df = pd.read_csv('Preprocessed_Data.csv')

# Check for missing values and drop them
print(f"Original shape: {df.shape}")
df = df.dropna()
print(f"After dropping NaN: {df.shape}")

# Quick text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

print("Cleaning text...")
# Apply cleaning
df['Cleaned_Text'] = df['Text'].apply(clean_text)

# Encode categories
print("Encoding categories...")
label_encoder = LabelEncoder()
df['Category_Encoded'] = label_encoder.fit_transform(df['Category'])

print("Categories found:", label_encoder.classes_)

# Use a more efficient TF-IDF vectorizer
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

X = tfidf.fit_transform(df['Cleaned_Text'])
y = df['Category_Encoded']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Save the vectorizer and encoder
joblib.dump(tfidf, 'tfidf.pkl')
joblib.dump(label_encoder, 'label.pkl')
print("Vectorizer and encoder saved")

# Model 1: Logistic Regression (Fast and effective for text)
print("Training Logistic Regression...")
lr = LogisticRegression(
    C=1.0, 
    solver='liblinear', 
    max_iter=1000,
    random_state=42
)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Model 2: Linear SVM (Fast and accurate)
print("Training SVM...")
svm = LinearSVC(
    C=0.5, 
    max_iter=2000, 
    random_state=42,
    dual=False
)
svm.fit(X_train, y_train)

# Calibrate SVM for probability estimates
calibrated_svm = CalibratedClassifierCV(svm, cv=3)  # Reduced CV for speed
calibrated_svm.fit(X_train, y_train)
y_pred_svm = calibrated_svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# Save the best model
if lr_accuracy > svm_accuracy:
    joblib.dump(lr, 'classifier_resume.pkl')
    best_model = lr
    best_accuracy = lr_accuracy
    print("Logistic Regression model saved as best")
else:
    joblib.dump(calibrated_svm, 'classifier_resume.pkl')
    best_model = calibrated_svm
    best_accuracy = svm_accuracy
    print("SVM model saved as best")

# Evaluation
print("\n=== FINAL EVALUATION ===")
print(f"Best Model Accuracy: {best_accuracy:.4f}")

y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as confusion_matrix.png")

print("Training completed successfully!")