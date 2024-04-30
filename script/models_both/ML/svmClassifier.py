import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
data_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack_data/script/features_extraction/trimmed_01s/IO_trimmed.csv'
data = pd.read_csv(data_path)

# Replace infinities with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Optional: Check for and handle very large values here, if applicable

# Drop rows with NaN values or impute
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data.drop('label', axis=1))
data_imputed = pd.DataFrame(data_imputed, columns=data.drop('label', axis=1).columns)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['label'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data_imputed, y_encoded, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train the SVM classifier
model = SVC(kernel='rbf', C=1, gamma='auto')
model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
