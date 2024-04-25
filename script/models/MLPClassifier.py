import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import multiprocessing

def train_model(params):
    num_units, dropout_rate, data_path = params
    # Load and preprocess data inside the function
    data = pd.read_csv(data_path)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.dropna(inplace=True)
    X = data.drop('label', axis=1).values
    y = data['label'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    fold_no = 1
    
    for train, test in kfold.split(X_scaled, y_encoded):
        print(f"Starting training for configuration: {num_units} units, {dropout_rate} dropout rate, fold {fold_no}")
        model = Sequential([
            Dense(num_units, activation='relu', input_shape=(X_scaled.shape[1],)),
            Dropout(dropout_rate),
            Dense(num_units // 2, activation='relu'),
            Dropout(dropout_rate),
            Dense(y_categorical.shape[1], activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_scaled[train], y_categorical[train], epochs=50, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_scaled[test], y_categorical[test], verbose=0)
        accuracies.append(accuracy * 100)  # Store accuracy as percentage
        print(f"Finished fold {fold_no} with accuracy: {accuracy * 100:.2f}%")
        fold_no += 1

    return num_units, dropout_rate, accuracies, np.mean(accuracies)

if __name__ == '__main__':
    data_path = '../features_extraction/IO.csv'
    configurations = [(128, 0.5, data_path), (128, 0.3, data_path), (256, 0.5, data_path), (256, 0.3, data_path)]

    # Prepare arguments for multiprocessing
    args = [(units, dropout, data_path) for units, dropout in [(128, 0.5), (128, 0.3), (256, 0.5), (256, 0.3)]]

    # Run in parallel
    with multiprocessing.Pool(processes=len(args)) as pool:
        results = pool.map(train_model, args)

    # Structure the results and save them to a CSV file
    results_df = pd.DataFrame(results, columns=['Num_Units', 'Dropout_Rate', 'Accuracies', 'Average_Accuracy'])
    results_df['Accuracies'] = results_df['Accuracies'].apply(lambda x: str(x))
    results_df.to_csv('mlp_cv_results.csv', index=False)
    print("Cross-validation results with configurations saved to 'mlp_cv_results.csv'.")
    print(results_df)
