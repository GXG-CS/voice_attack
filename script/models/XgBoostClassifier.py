from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np

# Load your dataset
data_path = '../../data_processed/WiSec_unmonitored_trimmed_5_features2.csv'
data = pd.read_csv(data_path)

# Handle infinities by replacing them with NaNs
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Prepare the features and target
X = data.drop('label', axis=1)
y = data['label']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define a preprocessing pipeline
preprocessing_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler()),  # Scale features
])

# Apply the preprocessing pipeline to the training data
X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)

# Apply the same preprocessing to the test data
X_test_preprocessed = preprocessing_pipeline.transform(X_test)

# Initialize XGBoost classifier
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

# Train the model
model.fit(X_train_preprocessed, y_train)

# Make predictions
y_pred = model.predict(X_test_preprocessed)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
