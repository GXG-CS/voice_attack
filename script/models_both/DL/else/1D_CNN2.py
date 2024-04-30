import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Load and preprocess dataset
data_path = '../../../data_processed/WiSec_monitored_trimmed_5_removeOutliers_features_filter.csv'  
data = pd.read_csv(data_path)

# Replace infinite values with NaNs and drop them
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# Separate features and labels
X = data.drop('label', axis=1).values
y = data['label'].values

# Encode labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)  # Convert to one-hot encoding

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define a function to create the model
def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Initialize K-Fold Cross-validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
scores = []

for train, test in kfold.split(X_scaled, y_encoded):
    model = create_model((X_scaled.shape[1], 1), y_categorical.shape[1])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Reshape data for CNN input
    X_train_reshaped = np.reshape(X_scaled[train], (X_scaled[train].shape[0], X_scaled[train].shape[1], 1))
    X_test_reshaped = np.reshape(X_scaled[test], (X_scaled[test].shape[0], X_scaled[test].shape[1], 1))
    
    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=1, mode='min', min_lr=0.00001)
    
    print(f'Training for fold {fold_no}...')
    history = model.fit(X_train_reshaped, y_categorical[train], batch_size=32, epochs=200, validation_split=0.2, verbose=2, callbacks=[early_stopping, reduce_lr])
    
    scores.append(model.evaluate(X_test_reshaped, y_categorical[test], verbose=0)[1])
    fold_no += 1

print(f'Average Test Accuracy across all folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}')
