import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, matthews_corrcoef
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np

# Load the dataset
data = pd.read_csv('/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/features_extraction/IO.csv')

data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

X = data.drop('label', axis=1).values  # Features
y = data['label'].values  # Labels

# Ensure labels start at 0 and are continuous
labels, unique = pd.factorize(y)

# Convert labels to categorical one-hot encoding
y_categorical = to_categorical(labels)

# Define 5-fold cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X, labels):

    # Define the model architecture
    model = Sequential()
    model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))  # Adjust the number of neurons
    model.add(Dense(32, activation='relu'))  # Add or remove layers as needed
    model.add(Dense(len(unique), activation='softmax'))  # Output layer with softmax activation

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Generate a print
    print(f'Training for fold {fold_no} ...')

    # Fit data to model
    history = model.fit(X[train], y_categorical[train], batch_size=32, epochs=50, verbose=1)

    # Predict on the test data
    y_pred_prob = model.predict(X[test])
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = labels[test]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    # Print metrics
    print(f'Fold {fold_no}: Accuracy: {accuracy*100}%, MCC: {mcc}')
    
    fold_no += 1

# Save the model if needed
# model.save('/Users/xiaoguang_guo@mines.edu/Documents/voice_attack/script/models/DL/model.h5')
