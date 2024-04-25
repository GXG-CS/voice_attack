import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from concurrent.futures import ThreadPoolExecutor

from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
# from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def top_k_accuracy_score(y_true, y_score, k=5):
    top_k = np.argsort(y_score, axis=1)[:, -k:]
    return np.mean([1 if y_true[i] in top_k[i] else 0 for i in range(len(y_true))])

def process_file(data_path):
    print(f"Processing {data_path}...")
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
    ])

    pipeline.fit(X_train, y_train)
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # Add additional classifiers like AdaBoost and GradientBoosting
    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'ExtraTrees': ExtraTreesClassifier(random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        # 'AdaBoost': AdaBoostClassifier(random_state=42),
        # 'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'MLPClassifier': MLPClassifier(random_state=42),
        # 'Bagging': BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'NaiveBayes': GaussianNB(),
        # 'LGBM': LGBMClassifier(random_state=42),
        # 'CatBoost': CatBoostClassifier(verbose=0, random_state=42)  # 'verbose=0' to turn off training verbosity

    }

    results = []
    for name, classifier in classifiers.items():
        print(f"Starting training for {name}...")
        classifier.fit(X_train_transformed, y_train)
        print(f"Training completed for {name}.")

        # Predict probabilities for the top-k accuracy if the classifier supports it
        if hasattr(classifier, "predict_proba"):
            y_probs = classifier.predict_proba(X_test_transformed)
            top_5_acc = top_k_accuracy_score(y_test, y_probs, k=5)
        else:
            top_5_acc = None  # If the classifier does not support probability estimates

        y_pred = classifier.predict(X_test_transformed)
        
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        print(f"Results for {name}:")
        print(f"Top-5 Accuracy: {top_5_acc}")
        print(f"Accuracy: {acc}")
        print(f"MCC: {mcc}")
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print("----------------------------------------------------")
        
        results.append({
            'Dataset Category': data_path.split('/')[-1].replace('.csv', ''),
            'Model': name,
            'Top-5 Accuracy': top_5_acc,
            'Accuracy': acc,
            'MCC': mcc,
            'F1 Score': f1,
            'Precision': precision,
            'Recall': recall
        })
    return results

# Paths to CSV files
data_paths = [
    # '../features_extraction/incoming.csv',
    # '../features_extraction/outgoing.csv',
    # '../features_extraction/raw.csv',
    '../features_extraction/IO.csv',
]

# Use ThreadPoolExecutor to run processing in parallel
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(process_file, path) for path in data_paths]
    results = [future.result() for future in futures]

# Flatten the list of results and convert to DataFrame
flat_results = [item for sublist in results for item in sublist]
results_df = pd.DataFrame(flat_results)
results_df.to_csv('model_evaluation_results.csv', index=False)



