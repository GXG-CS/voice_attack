import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Load your dataset
data_path = '../../features_extraction/IO.csv'
data = pd.read_csv(data_path)

print("Data Loaded Successfully. Data shape:", data.shape)

# Replace infinities with NaN and drop rows with NaN values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

print("Data after cleaning. Data shape:", data.shape)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['label'])
X = data.drop('label', axis=1)

print("Features and labels prepared. Feature shape:", X.shape, "Label shape:", y_encoded.shape)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Data split into training and testing. Training shape:", X_train.shape, "Testing shape:", X_test.shape)

# Define a preprocessing pipeline
pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('smote', SMOTE(random_state=42)),            # Handle class imbalance
    ('scaler', StandardScaler()),                 # Scale features
    ('pca', PCA(n_components=25)),                # PCA for dimensionality reduction
])

# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)

# Transform both training and test sets
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)

# Initialize XGBoost classifier
classifier = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
classifier.fit(X_train_transformed, y_train)

# Evaluate the best model on the test set
y_pred = classifier.predict(X_test_transformed)

# Classification report
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)
print("Classification Report on test set:")
print(classification_report(y_test, y_pred))

