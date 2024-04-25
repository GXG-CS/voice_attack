import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def top_k_accuracy_score(y_true, y_score, k=5):
    top_k = np.argsort(y_score, axis=1)[:, -k:]
    return np.mean([1 if y_true[i] in top_k[i] else 0 for i in range(len(y_true))])

data_path = '../../../data_processed/vc_200/alexa/combined_features.csv'
data = pd.read_csv(data_path)

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['label'])
X = data.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

pipeline = ImbPipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('smote', SMOTE(random_state=42)),
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', solver='lbfgs'))
])

pipeline.fit(X_train, y_train)
y_probs = pipeline.predict_proba(X_test)

top_5_accuracy = top_k_accuracy_score(y_test, y_probs, k=5)
print(f"Top-5 Accuracy: {top_5_accuracy * 100:.2f}%")

y_pred = pipeline.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
