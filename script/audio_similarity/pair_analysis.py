import librosa
import numpy as np
from scipy.spatial.distance import cosine

def extract_mfcc_features(audio_path):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(audio_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    return mfccs

def calculate_similarity(mfccs1, mfccs2):
    """Calculate cosine similarity between two sets of MFCC features."""
    # Flatten MFCC coefficients
    mfccs1_flat = mfccs1.flatten()
    mfccs2_flat = mfccs2.flatten()
    
    # Ensure same length by padding shorter sequence with zeros
    max_len = max(len(mfccs1_flat), len(mfccs2_flat))
    mfccs1_flat = np.pad(mfccs1_flat, (0, max_len - len(mfccs1_flat)), 'constant')
    mfccs2_flat = np.pad(mfccs2_flat, (0, max_len - len(mfccs2_flat)), 'constant')
    
    # Calculate cosine similarity (1 means identical, 0 means orthogonal)
    similarity = 1 - cosine(mfccs1_flat, mfccs2_flat)
    return similarity

# Paths to your audio files
audio_path_1 = '../../raw_data/WiSec/monitored_100/audioPlay_A/simple_1.wav'
audio_path_2 = '../../raw_data/WiSec/monitored_100/audioPlay_A/simple_2.wav'

# Extract MFCC features
mfccs1 = extract_mfcc_features(audio_path_1)
mfccs2 = extract_mfcc_features(audio_path_2)

# Calculate similarity
similarity = calculate_similarity(mfccs1, mfccs2)
print(f"Similarity: {similarity}")
