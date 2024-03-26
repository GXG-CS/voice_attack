import numpy as np
import librosa
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

# Choose the number of clusters
n_clusters = 5

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=10)
cluster_labels = kmeans.fit_predict(features_normalized)

# Calculate the silhouette score for each sample
silhouette_avg = silhouette_score(features_normalized, cluster_labels)
print(f"Silhouette score for {n_clusters} clusters: {silhouette_avg}")

# Calculate silhouette scores for each sample
sample_silhouette_values = silhouette_samples(features_normalized, cluster_labels)

# Plotting
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(18, 7)

y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    y_lower = y_upper + 10

ax1.set_title("Silhouette plot for the various clusters.")
ax1.set_xlabel("Silhouette coefficient values")
ax1.set_ylabel("Cluster label")

# The vertical line for average silhouette score of all the samples
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
             fontsize=14, fontweight='bold')

plt.show()
