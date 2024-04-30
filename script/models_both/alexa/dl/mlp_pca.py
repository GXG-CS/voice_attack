import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import multiprocessing

def train_model(params):
    num_filters, kernel_size, activation, dropout_rate, learning_rate, data_path = params
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
    X_reshaped = np.reshape(X_pca, (X_pca.shape[0], X_pca.shape[1], 1))

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)
    
    model = Sequential([
        Conv1D(num_filters, kernel_size=kernel_size, activation=activation, input_shape=(X_pca.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Conv1D(num_filters * 2, kernel_size=kernel_size, activation=activation),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(128, activation=activation),
        Dropout(dropout_rate),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)

    return num_filters, kernel_size, activation, dropout_rate, learning_rate, accuracy * 100

if __name__ == '__main__':
    data_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/features_extraction_both/alexa/no_trim/IO.csv'
    configurations = [
        # (32, 3, 'relu', 0.3, 0.001, data_path),
        # (64, 3, 'relu', 0.5, 0.001, data_path),
        (64, 5, 'tanh', 0.3, 0.001, data_path),
        # (128, 3, 'sigmoid', 0.5, 0.01, data_path)
    ]

    with multiprocessing.Pool(processes=len(configurations)) as pool:
        results = pool.map(train_model, configurations)

    results_df = pd.DataFrame(results, columns=['Num_Filters', 'Kernel_Size', 'Activation', 'Dropout_Rate', 'Learning_Rate', 'Test_Accuracy'])
    results_df.to_csv('cnn_25_results_100.csv', index=False)
    print("Test results saved to 'cnn_test_results.csv'.")
    print(results_df)
