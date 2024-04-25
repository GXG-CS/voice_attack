# import pandas as pd
# import numpy as np
# from sklearn.model_selection import StratifiedKFold
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.callbacks import EarlyStopping

# # Load dataset
# data_path = '../features_extraction/IO.csv'
# data = pd.read_csv(data_path)

# # Handle infinite values and drop rows with NaN
# data.replace([np.inf, -np.inf], np.nan, inplace=True)
# data.dropna(inplace=True)

# # Separate features and label
# X = data.drop('label', axis=1).values
# y = data['label'].values

# # Encode labels
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# y_categorical = to_categorical(y_encoded)

# # Feature scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # Reshape input to be [samples, time steps, features]
# X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# # Define the LSTM model
# def create_model(input_shape, num_classes, lstm_units):
#     model = Sequential([
#         LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
#         Dropout(0.2),
#         LSTM(lstm_units // 2),
#         Dropout(0.2),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Prepare for 5-fold cross-validation
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# results = []

# # Configuration search
# lstm_configurations = [ 64, 128, 256]  # Different LSTM configurations
# for lstm_units in lstm_configurations:
#     acc_scores = []
#     fold_no = 1
#     for train, test in kfold.split(X_reshaped, y_encoded):
#         model = create_model((1, X_reshaped.shape[2]), y_categorical.shape[1], lstm_units)
#         print(f'Training fold {fold_no} with {lstm_units} LSTM units...')
        
#         # Fit the model
#         history = model.fit(X_reshaped[train], y_categorical[train], epochs=1, batch_size=32,
#                             validation_data=(X_reshaped[test], y_categorical[test]), verbose=2,
#                             callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
        
#         # Evaluate the model
#         scores = model.evaluate(X_reshaped[test], y_categorical[test], verbose=0)
#         print(f'Score for fold {fold_no}: Accuracy of {scores[1]*100:.2f}%')
#         acc_scores.append(scores[1] * 100)
#         fold_no += 1
    
#     # Record the results
#     results.append({
#         'LSTM Units': lstm_units,
#         'Accuracies': acc_scores,
#         'Average Accuracy': np.mean(acc_scores)
#     })

# # Save the results
# results_df = pd.DataFrame(results)
# results_df.to_csv('lstm_cv_results_configured.csv', index=False)
# print("Cross-validation results with configurations saved to 'lstm_cv_results_configured.csv'.")
# print(results_df)


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import multiprocessing

# Define the LSTM model
def create_model(input_shape, num_classes, lstm_units):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(lstm_units // 2),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to create and train a model
def train_and_evaluate(args):
    lstm_units, data_path = args  # Unpack arguments
    # Load and prepare data inside the function
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
        model = create_model((1, X_reshaped.shape[2]), y_categorical.shape[1], lstm_units)
        model.fit(X_reshaped[train], y_categorical[train], epochs=100, batch_size=32,
                  validation_data=(X_reshaped[test], y_categorical[test]), verbose=2,
                  callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
        scores = model.evaluate(X_reshaped[test], y_categorical[test], verbose=0)
        acc_scores.append(scores[1] * 100)

    return {
        'LSTM Units': lstm_units,
        'Accuracies': acc_scores,
        'Average Accuracy': np.mean(acc_scores)
    }

if __name__ == '__main__':
    # Data path and LSTM configurations
    data_path = '../features_extraction/IO.csv'
    lstm_configurations = [64, 128, 256]

    # Prepare arguments for multiprocessing
    args = [(units, data_path) for units in lstm_configurations]

    # Parallel processing
    with multiprocessing.Pool(processes=len(lstm_configurations)) as pool:
        results = pool.map(train_and_evaluate, args)

    # Save the results
    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_cv_results_configured.csv', index=False)
    print("Cross-validation results with configurations saved to 'lstm_cv_results_configured.csv'.")
    print(results_df)

