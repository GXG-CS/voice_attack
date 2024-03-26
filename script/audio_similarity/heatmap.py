import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from glob import glob

def extract_mfcc_features(audio_path, n_mfcc=20):
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Load audio files
audio_files = glob('../../raw_data/WiSec/unmonitored_100/audioPlay_A/*.wav')[:200]  # Adjust path, limit to 200 files

# Extract features from each audio file
features = np.array([extract_mfcc_features(file) for file in audio_files])

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Compute the similarity matrix
similarity_matrix = cosine_similarity(features_normalized)

# Visualize the similarity matrix
plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, cmap='viridis')
plt.title('Audio File Similarity Matrix')
plt.xlabel('Audio File Index')
plt.ylabel('Audio File Index')
plt.show()
