import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

def top_k_accuracy_score(y_true, y_probs, k=5):
    """Calculate top-k accuracy classification score."""
    top_k = np.argsort(y_probs, axis=1)[:, -k:]
    return np.mean([1 if y_true[i] in top_k[i] else 0 for i in range(len(y_true))])

# Load dataset
data_path = '../../../data_processed/vc_200/alexa/combined_features.csv'
data = pd.read_csv(data_path)
print("Data loaded successfully. Number of instances:", len(data))

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)
print("Data cleaned. Remaining instances after removing NAs:", len(data))

# Prepare data
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['label'])
X = data.drop('label', axis=1)
print("Data encoded. Features:", X.shape[1], "Labels:", len(np.unique(y_encoded)))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print("Data split into train and test sets. Train size:", len(X_train), "Test size:", len(X_test))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Features scaled.")

# Initialize and train classifier
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
print("Training Gradient Boosting Classifier...")
classifier.fit(X_train_scaled, y_train)
print("Model training completed.")

# Predict probabilities
y_probs = classifier.predict_proba(X_test_scaled)
print("Predictions completed.")

# Calculate top-5 accuracy
top_5_accuracy = top_k_accuracy_score(y_test, y_probs, k=5)
print(f"Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")

# Regular accuracy and classification report
y_pred = np.argmax(y_probs, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print("Regular Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))
