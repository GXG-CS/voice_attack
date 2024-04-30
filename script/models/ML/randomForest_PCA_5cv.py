import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the data
data_path = '../features_extraction/IO.csv'
data = pd.read_csv(data_path)

# Check and handle infinite values and very large values
data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities with NaN
data.dropna(inplace=True)  # Drop rows with NaN values, which now includes former infinities

# Separate features and target variable
X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable

# Initialize the StandardScaler
scaler = StandardScaler()

# Initialize PCA with 66 components
pca = PCA(n_components=66)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Prepare for cross-validation
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

# Perform 5-fold cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Scale data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Train the model
    model.fit(X_train_pca, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_pca)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Append results
    results.append({
        'Fold': len(results) + 1,
        'Accuracy': accuracy,
        'Confusion Matrix': str(conf_matrix),
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score']
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save results to CSV
results_df.to_csv('cv_results.csv', index=False)

print("Cross-validation results saved to 'cv_results.csv'.")
print(results_df)
