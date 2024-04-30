from scapy.all import rdpcap
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch
from numpy.fft import fft

def parse_pcap(file_path):
    """Parse a pcap file to extract packet sizes and inter-arrival times."""
    packets = rdpcap(file_path)
    sizes = np.array([len(p) for p in packets])
    times = np.array([p.time for p in packets])
    inter_arrival_times = np.diff(times) if len(times) > 1 else np.array([0])
    return sizes, inter_arrival_times

def calculate_basic_features(data):
    """Calculate basic statistical features from data."""
    mean = np.mean(data)
    std_dev = np.std(data)
    variance = np.var(data)
    max_value = np.max(data)
    min_value = np.min(data)
    range_value = max_value - min_value
    median = np.median(data)
    mad = np.mean(np.abs(data - mean))  # Mean absolute deviation
    skewness = skew(data)
    kurt = kurtosis(data)
    return {
        'mean': mean,
        'std_dev': std_dev,
        'variance': variance,
        'max_value': max_value,
        'min_value': min_value,
        'range': range_value,
        'median': median,
        'mad': mad,
        'skewness': skewness,
        'kurtosis': kurt,
    }

def calculate_fft_features(data):
    """Calculate FFT based features from data."""
    fft_values = np.abs(fft(data))
    fft_mean = np.mean(fft_values)
    fft_std_dev = np.std(fft_values)
    return {
        'fft_mean': fft_mean,
        'fft_std_dev': fft_std_dev,
    }

def calculate_entropy(data):
    """Calculate entropy of the data."""
    _, counts = np.unique(data, return_counts=True)
    return {'entropy': entropy(counts)}

def calculate_peak_features(data):
    """Calculate features based on peaks in the data."""
    peaks, _ = find_peaks(data)
    num_peaks = len(peaks)
    mean_peak_height = np.mean(data[peaks]) if num_peaks > 0 else 0
    return {
        'num_peaks': num_peaks,
        'mean_peak_height': mean_peak_height,
    }

def calculate_signal_features(sizes, inter_arrival_times):
    """Calculate signal processing features from data."""
    rms = np.sqrt(np.mean(sizes**2))  # Root Mean Square
    sma = np.sum(np.abs(sizes)) / len(sizes)  # Signal Magnitude Area
    autocorr_lag_1 = np.corrcoef(sizes[:-1], sizes[1:])[0, 1] if len(sizes) > 1 else 0  # Autocorrelation
    zero_crossing_rate = ((sizes[:-1] * sizes[1:]) < 0).sum() / len(sizes)
    spectral_centroid = np.sum(np.arange(len(sizes)) * sizes) / np.sum(sizes) if np.sum(sizes) > 0 else 0
    return {
        'rms': rms,
        'sma': sma,
        'autocorrelation_lag_1': autocorr_lag_1,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid': spectral_centroid,
    }

def extract_features_from_pcap(file_path):
    """Extract and combine all features from a pcap file."""
    sizes, inter_arrival_times = parse_pcap(file_path)
    features = {}
    features.update(calculate_basic_features(sizes))
    features.update(calculate_fft_features(sizes))
    features.update(calculate_entropy(sizes))
    features.update(calculate_peak_features(sizes))
    features.update(calculate_signal_features(sizes, inter_arrival_times))
    return features

# Example usage
file_path = '../../data_processed/traffic_W_outgoing/1/2_outgoing.pcap'
features = extract_features_from_pcap(file_path)
print(features)
