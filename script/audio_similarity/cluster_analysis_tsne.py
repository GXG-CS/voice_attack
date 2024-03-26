import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from glob import glob

def extract_mfcc_features(audio_path, n_mfcc=20):
    try:
        y, sr = librosa.load(audio_path)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

# Update this path to where your audio files are located
audio_files = glob('../../raw_data/WiSec/unmonitored_100/audioPlay_A/*.wav')
if not audio_files:
    raise ValueError("No audio files found. Check the path.")

# Extract features from each audio file
features = []
for file in audio_files:
    feature = extract_mfcc_features(file)
    if feature is not None:
        features.append(feature)
features = np.array(features)

if features.size == 0:
    raise ValueError("No features extracted. Check the audio files and feature extraction process.")

# Normalize the features
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Apply K-means clustering
n_clusters = 5  # Adjust this based on your specific needs
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(features_normalized)

# Apply t-SNE for dimensionality reduction to 2D for visualization
tsne = TSNE(n_components=2, perplexity=30, n_iter=3000, random_state=0)
features_reduced = tsne.fit_transform(features_normalized)

# Plotting
plt.figure(figsize=(10, 6))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
for i in range(n_clusters):
    cluster_indices = np.where(kmeans.labels_ == i)[0]
    plt.scatter(features_reduced[cluster_indices, 0], features_reduced[cluster_indices, 1], 
                label=f'Cluster {i}', color=colors[i % len(colors)])
plt.legend()
plt.title('Voice Command Clusters after t-SNE Reduction')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()
