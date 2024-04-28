import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Function to calculate top-k accuracy
def top_k_accuracy_score(y_true, y_score, k=5):
    """Compute top-k accuracy score. Returns the top-k accuracy score."""
    top_k = np.argsort(y_score, axis=1)[:, -k:]
    return np.mean([1 if y_true[i] in top_k[i] else 0 for i in range(len(y_true))])

# Load your dataset
data_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/features_extraction/IO.csv'
data = pd.read_csv(data_path)

# Replace infinities with NaN and drop rows with NaN values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['label'])
X = data.drop('label', axis=1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define a preprocessing pipeline
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('smote', SMOTE(random_state=42)),            # Handle class imbalance
    ('scaler', StandardScaler()),                 # Scale features
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Transform both training and test sets
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Initialize XGBoost classifier
classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
classifier.fit(X_train_transformed, y_train)

# Predict probabilities on the test set
y_probs = classifier.predict_proba(X_test_transformed)

# Calculate top-5 accuracy
top_5_accuracy = top_k_accuracy_score(y_test, y_probs, k=5)
print(f"Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")

# Regular accuracy for comparison
y_pred = np.argmax(y_probs, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Regular Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
