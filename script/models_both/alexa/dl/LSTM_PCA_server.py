import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
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

    pca = PCA(n_components=25)
    X_pca = pca.fit_transform(X_scaled)

    X_reshaped = np.reshape(X_pca, (X_pca.shape[0], 1, X_pca.shape[1]))

    # Create the model using the entire dataset
    model = create_model((1, X_reshaped.shape[2]), y_categorical.shape[1], lstm_units, dropout_rate, learning_rate)
    model.fit(X_reshaped, y_categorical, epochs=200, batch_size=32, verbose=2, callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)])
    _, accuracy = model.evaluate(X_reshaped, y_categorical, verbose=0)

    return {
        'LSTM Units': lstm_units,
        'Dropout Rate': dropout_rate,
        'Learning Rate': learning_rate,
        'Accuracy': accuracy * 100
    }

if __name__ == '__main__':
    data_path = '../../../features_extraction_both/alexa/no_trim/IO_01s.csv'
    configurations = [
        (256, 0.3, 0.001, data_path)
    ]

    with multiprocessing.Pool(processes=min(len(configurations), multiprocessing.cpu_count())) as pool:
        results = pool.map(train_and_evaluate, configurations)

    results_df = pd.DataFrame(results)
    results_df.to_csv('lstm_25_01s_200.csv', index=False)
    print("Training results saved to 'lstm_results_25.csv'.")
    print(results_df)
