import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

data = pd.read_csv('/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/features_extraction/IO.csv')
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(data['label'])
y_categorical = to_categorical(y_encoded)
X = data.drop('label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=1)

y_probs = model.predict(X_test_scaled)
top_k_accuracy = np.mean(np.any(y_test[:, np.argsort(y_probs)[:, -5:]], axis=1))
print(f"Top-5 Accuracy: {top_k_accuracy * 100:.2f}%")

# Evaluate the model
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print("Accuracy:", accuracy)
