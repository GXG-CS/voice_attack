import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import numpy as np

# Function to load data, preprocess, and split
def prepare_data(data_path):
    print(f"Loading and preprocessing data from: {data_path}")
    data = pd.read_csv(data_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data['label'])
    X = data.drop('label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42)
    
    print("Data preparation complete.")
    return X_train, X_test, y_train, y_test

# Define a Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=1000, random_state=42))
])

# Parameter grid for GridSearchCV
parameter_space = {
    'mlp__hidden_layer_sizes': [(50,), (100,), (50,50), (100,100)],
    'mlp__activation': ['tanh', 'relu'],
    'mlp__solver': ['sgd', 'adam'],
    'mlp__alpha': [0.0001, 0.05],
    'mlp__learning_rate': ['constant', 'adaptive'],
}

# Load data
X_train, X_test, y_train, y_test = prepare_data('../features_extraction/IO.csv')

# Perform the grid search with 5-fold cross-validation and more verbose output
def run_grid_search(X_train, y_train):
    grid_search = GridSearchCV(pipeline, parameter_space, n_jobs=-1, cv=5, return_train_score=False)
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    print("Grid search complete.")
    return grid_search

# Running the updated grid search
grid_search = run_grid_search(X_train, y_train)
print('Best parameters found:\n', grid_search.best_params_)

# Print intermediate results
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, std, param in zip(means, stds, params):
    print(f"Evaluated: {param}")
    print(f"Mean test score: {mean:.3f} (Std: {std:.3f})")

# Save the grid search results to a csv file
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv('grid_search_results.csv', index=False)

# Predict and print the classification report for the best estimator
y_pred = grid_search.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('classification_report.csv', index=True)

print(report_df)
