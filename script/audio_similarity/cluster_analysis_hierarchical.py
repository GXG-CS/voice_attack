import numpy as np
import librosa
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from glob import glob

def extract_mfcc_features(audio_path, n_mfcc=20):
    try:
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Load audio files
audio_files = glob('../../raw_data/WiSec/monitored_100/audioPlay_A/*.wav')
if not audio_files:
    raise ValueError("No audio files found. Check the path.")

# Extract features
features = [extract_mfcc_features(file) for file in audio_files]
features = [f for f in features if f is not None]  # Remove None values
if not features:
    raise ValueError("No features extracted. Check the audio files and feature extraction process.")
features = np.array(features)

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Perform hierarchical clustering
linked = linkage(features_normalized, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           labels=[f.split('/')[-1] for f in audio_files],
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index or (Cluster size)')
plt.ylabel('Distance')
plt.show()
