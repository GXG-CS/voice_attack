import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import numpy as np

# Load dataset
# data_path = '../../data_processed/WiSec_unmonitored_trimmed_5_features2.csv'
data_path = '/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/features_extraction/IO.csv'
data = pd.read_csv(data_path)

# Handle infinite values
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# # Identify labels to remove (hypothetical example labels)
# low_acc_labels = [5, 8, 12, 14, 16, 17, 20, 34, 4, 7, 10, 15, 21, 26, 30, 35, 36, 50, 96, 97, 100, 6, 10, 15, 19, 21, 25, 28, 30, 31, 32, 35, 36]

# # Filter out rows with low accuracy labels
# data_filtered = data[~data['label'].isin(low_acc_labels)]


# Separate features and label
X = data.drop('label', axis=1).values
y = data['label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape input to be [samples, time steps, features]
# Here, considering each sample as a sequence of one time step with multiple features
X_reshaped = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

# Splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_categorical, test_size=0.2, random_state=42)

# Define an adjusted LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, X_train.shape[2])))  # Increased units
model.add(Dropout(0.2))  # Dropout for regularization
model.add(LSTM(64))  # Additional LSTM layer
model.add(Dropout(0.2))  # Additional dropout layer
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model with an adjustable learning rate
optimizer = Adam(learning_rate=0.001)  # Consider tuning the learning rate
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Adjusted training with verbose output to monitor progress
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=2)
# Evaluate the model
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Test accuracy: {accuracy:.4f}')
