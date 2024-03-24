import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the data
data_path = '../../data_processed/WiSec_unmonitored_trimmed_5_features2.csv'
data = pd.read_csv(data_path)

# Check and handle infinite values and very large values
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
data.dropna(inplace=True)  # Drop rows with NaN values, which now includes former infinities

# Identify labels to remove (hypothetical example labels)
low_acc_labels = [5, 8, 12, 14, 16, 17, 20, 34, 4, 7, 10, 15, 21, 26, 30, 35, 36, 50, 96, 97, 100, 6, 10, 15, 19, 21, 25, 28, 30, 31, 32, 35, 36]

# Filter out rows with low accuracy labels
data_filtered = data[~data['label'].isin(low_acc_labels)]

# Separate features and target variable after filtering
X = data_filtered.drop('label', axis=1)  # Features
y = data_filtered['label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Predict on the scaled test set
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
