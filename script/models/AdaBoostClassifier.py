import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer 

# Load the data
data_path = '../features_extraction/combined_incoming_outgoing.csv'  
data = pd.read_csv(data_path)

# Separate features and target variable
X = data.drop('label', axis=1)  # Features
y = data['label']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Imputer to handle NaN values
imputer = SimpleImputer(strategy='mean')  # Can change strategy to 'median' or 'most_frequent' if more appropriate

# Apply imputation to fill in NaN values
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# It's generally a good practice to scale features, especially for linear models, but it's optional for tree-based models like AdaBoost. 
# Initialize and apply the StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Initialize the AdaBoostClassifier model
model = AdaBoostClassifier(n_estimators=100, random_state=42)

# Train the model on the scaled and imputed training data
model.fit(X_train_scaled, y_train)

# Predict on the scaled and imputed test data
y_pred = model.predict(X_test_scaled)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
