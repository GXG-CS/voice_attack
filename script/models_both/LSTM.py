import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing

def create_model(input_shape, num_classes, lstm_units, dropout_rate, learning_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units // 2),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(args):
    lstm_units, dropout_rate, learning_rate, data_path = args
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
    X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    
    for train, test in kfold.split(X_reshaped, y_encoded):
        model = create_model((1, X_reshaped.shape[2]), y_categorical.shape[1], lstm_units, dropout_rate, learning_rate)
        model.fit(X_reshaped[train], y_categorical[train], epochs=100, batch_size=32, validation_data=(X_reshaped[test], y_categorical[test]), verbose=2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
        _, accuracy = model.evaluate(X_reshaped[test], y_categorical[test], verbose=0)
        acc_scores.append(accuracy * 100)

    return {
        'LSTM Units': lstm_units,
        'Dropout Rate': dropout_rate,
        'Learning Rate': learning_rate,
        'Accuracies': acc_scores,
        'Average Accuracy': np.mean(acc_scores)
    }

if __name__ == '__main__':
    data_path = '../features_extraction/IO_1s.csv'
    configurations = [
        # (64, 0.2, 0.001, data_path),
        # (64, 0.3, 0.001, data_path),
        # (128, 0.2, 0.001, data_path),
        # (128, 0.3, 0.001, data_path),
        # (128, 0.2, 0.01, data_path),
        # (128, 0.3, 0.01, data_path),
        # (256, 0.2, 0.001, data_path),
        (256, 0.3, 0.001, data_path)
    ]

    with multiprocessing.Pool(processes=min(len(configurations), multiprocessing.cpu_count())) as pool:
        results = pool.map(train_and_evaluate, configurations)

    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_cv_results_1s.csv', index=False)
    print("Cross-validation results with extended configurations saved to 'lstm_cv_results_extended.csv'.")
    print(results_df)
