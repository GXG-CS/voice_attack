import numpy as np
import os
import pickle
import argparse
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch
from numpy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
import pywt

def extract_features(time_series):
    # Separate time and value components
    times, values = zip(*time_series)
    times = np.array(times)
    values = np.array(values)

    # Basic statistics
    mean = np.mean(values)
    std_dev = np.std(values, ddof=1)
    variance = np.var(values, ddof=1)
    max_value = np.max(values)
    min_value = np.min(values)
    range_value = max_value - min_value
    median = np.median(values)
    mad = np.mean(np.abs(values - mean))
    skewness = skew(values)
    kurt = kurtosis(values)

    # FFT-based features
    fft_vals = np.abs(fft(values))
    fft_mean = np.mean(fft_vals)
    fft_std_dev = np.std(fft_vals)

    # Entropy
    hist, _ = np.histogram(values, bins='auto')
    data_entropy = entropy(hist)

    # Peak-related features with adjustment
    peaks, properties = find_peaks(values, height=None)
    num_peaks = len(peaks)
    if num_peaks > 0 and 'peak_heights' in properties:
        mean_peak_height = np.mean(properties['peak_heights'])
    else:
        mean_peak_height = 0

    # Time difference features
    time_diffs = np.diff(times)
    mean_time_diff = np.mean(time_diffs)
    std_time_diff = np.std(time_diffs)

    # Advanced Features
    rms = np.sqrt(np.mean(np.square(values)))  # Root Mean Square
    sma = np.sum(np.abs(values)) / len(values)  # Signal Magnitude Area

    # Autocorrelation at lag 1
    autocorr_lag_1 = np.corrcoef(values[:-1], values[1:])[0, 1] if len(values) > 1 else np.nan

    # Crest factor
    crest_factor = max_value / rms if rms != 0 else np.nan

    # Zero crossing rate
    zero_crossing_rate = ((values[:-1] * values[1:]) < 0).sum() / float(len(values)-1) if len(values) > 1 else np.nan

    # Spectral centroid
    frequencies, power_spectrum = welch(values)
    spectral_centroid = np.sum(frequencies * power_spectrum) / np.sum(power_spectrum) if np.sum(power_spectrum) != 0 else np.nan

    return {
        'mean': mean,
        'std_dev': std_dev,
        'variance': variance,
        'max_value': max_value,
        'min_value': min_value,
        'range': range_value,
        'median': median,
        'mean_absolute_deviation': mad,
        'skewness': skewness,
        'kurtosis': kurt,
        'fft_mean': fft_mean,
        'fft_std_dev': fft_std_dev,
        'entropy': data_entropy,
        'num_peaks': num_peaks,
        'mean_peak_height': mean_peak_height,
        'mean_time_diff': mean_time_diff,
        'std_time_diff': std_time_diff,
        'rms': rms,
        'sma': sma,
        'autocorrelation_lag_1': autocorr_lag_1,
        'crest_factor': crest_factor,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid': spectral_centroid,
    }

def process_and_save_features(input_directory, output_directory):
    if not os.path.exists(input_directory):
        raise FileNotFoundError(f"Input directory does not exist: {input_directory}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith('.pkl'):
            input_file_path = os.path.join(input_directory, filename)
            with open(input_file_path, 'rb') as file:
                time_series_data = pickle.load(file)
            features_list = [extract_features(ts) for ts in time_series_data]
            output_file_path = os.path.join(output_directory, f'features_{filename}')
            with open(output_file_path, 'wb') as file:
                pickle.dump(features_list, file)
            print(f"Processed and saved features for {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from time series data.")
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input .pkl files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the extracted features .pkl files')
    args = parser.parse_args()
    process_and_save_features(args.input_dir, args.output_dir)
    print("Features have been extracted and saved successfully.")

# python features_extraction2.py --input_dir ../data_processed/traffic_W_outgoing_timeSeries_removeD_pickles --output_dir traffic_W_outgoing_features
# python features_extraction2.py --input_dir ../data_processed/traffic_W_incoming_timeSeries_removeD_pickles --output_dir traffic_W_incoming_features
