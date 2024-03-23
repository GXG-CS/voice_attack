import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks, welch
from numpy.fft import fft
from scapy.all import rdpcap
from glob import glob
from decimal import Decimal

def parse_pcap(file_path):
    packets = rdpcap(file_path)
    sizes = np.array([len(p) for p in packets]) if packets else np.array([])
    times = np.array([p.time for p in packets]) if packets else np.array([])
    inter_arrival_times = np.diff(times) if len(times) > 1 else np.array([])
    return sizes, times, inter_arrival_times


def calculate_features(sizes, times):
    if sizes.size == 0 or times.size == 0:
        return None
    
    start_time = Decimal(str(times[0]))
    end_time = Decimal(str(times[-1]))
    duration = end_time - start_time

    mean = Decimal(str(np.mean(sizes)))
    std_dev = Decimal(str(np.std(sizes, ddof=1)))
    variance = Decimal(str(np.var(sizes, ddof=1)))
    max_value = Decimal(str(np.max(sizes)))
    min_value = Decimal(str(np.min(sizes)))
    range_value = max_value - min_value
    median = Decimal(str(np.median(sizes)))
    mad = Decimal(str(np.mean(np.abs(sizes - np.mean(sizes)))))
    skewness = Decimal(str(skew(sizes)))
    kurt = Decimal(str(kurtosis(sizes)))

    fft_vals = np.abs(fft(sizes))
    fft_mean = Decimal(str(np.mean(fft_vals)))
    fft_std_dev = Decimal(str(np.std(fft_vals)))

    hist, _ = np.histogram(sizes, bins='auto')
    data_entropy = Decimal(str(entropy(hist)))

    peaks, properties = find_peaks(sizes)
    num_peaks = len(peaks)

    rms = Decimal(str(np.sqrt(np.mean(np.square(sizes)))))
    sma = Decimal(str(np.mean(np.abs(sizes))))
    
    if len(sizes) > 1:
        autocorr_lag_1 = Decimal(str(np.corrcoef(sizes[:-1], sizes[1:])[0, 1]))
    else:
        autocorr_lag_1 = Decimal(0)
    
    crest_factor = Decimal(str(np.max(np.abs(sizes)) / rms))
    zero_crossing_indices = np.where(np.diff(np.signbit(sizes)))[0]
    zero_crossing_rate = Decimal(str(len(zero_crossing_indices) / float(sizes.size)))
    
    spectral_centroid = Decimal(str(np.dot(np.arange(len(fft_vals)), fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0))

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
        'duration': duration, 
        'fft_mean': fft_mean,
        'fft_std_dev': fft_std_dev,
        'entropy': data_entropy,
        'num_peaks': num_peaks,
        'rms': rms,
        'sma': sma,
        'autocorrelation_lag_1': autocorr_lag_1,
        'crest_factor': crest_factor,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid': spectral_centroid,
    }


def extract_features_from_pcap(file_path):
    sizes, times, _ = parse_pcap(file_path)
    return calculate_features(sizes, times)

def process_file(file_path):
    print(f"Processing file: {file_path}")
    features, error = extract_features_from_pcap(file_path), None
    return features, error


def process_folder(folder_path, label, executor):
    print(f"Processing folder: {folder_path}, Label: {label}")
    features_list = []
    file_paths = sorted(glob(os.path.join(folder_path, f'simple_{label}', '*.pcap')))
    futures = [executor.submit(process_file, file_path) for file_path in file_paths]
    for future in futures:
        features, error = future.result()
        if features and not error:
            features_list.append(features)
    return features_list

def get_labels(folder_path):
    print(f"Getting labels from folder: {folder_path}")
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    labels = [subfolder.split('_')[-1] for subfolder in subfolders]
    labels = sorted(set(labels), key=int)
    print(f"Found labels: {labels}")
    return labels

def write_features_to_csv(combined_features, csv_file_path):
    print(f"Writing features to CSV: {csv_file_path}")
    if not combined_features:
        print("No combined features to write.")
        return
    with open(csv_file_path, 'w', newline='') as file:
        fieldnames = ['label'] + [f'{feat}_incoming' for feat in combined_features[0][1]] + [f'{feat}_outgoing' for feat in combined_features[0][1]]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label, inc_features, out_features in combined_features:
            row = {'label': label}
            row.update({f'{k}_incoming': v for k, v in inc_features.items()})
            row.update({f'{k}_outgoing': v for k, v in out_features.items()})
            writer.writerow(row)
    print("CSV writing complete.")

def main(incoming_folder, outgoing_folder, csv_file_path):
    labels = get_labels(incoming_folder)
    combined_features = []

    with ProcessPoolExecutor() as executor:
        for label in labels:
            print(f"Processing label: {label}")
            incoming_features = process_folder(incoming_folder, label, executor)
            outgoing_features = process_folder(outgoing_folder, label, executor)
            for inc_features, out_features in zip(incoming_features, outgoing_features):
                combined_features.append((label, inc_features, out_features))

    write_features_to_csv(combined_features, csv_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and combine .pcap file features.")
    parser.add_argument("--incoming_folder", required=True, help="Folder with incoming .pcap files.")
    parser.add_argument("--outgoing_folder", required=True, help="Folder with outgoing .pcap files.")
    parser.add_argument("--csv_file_path", required=True, help="CSV file path for the combined features.")
    args = parser.parse_args()

    main(args.incoming_folder, args.outgoing_folder, args.csv_file_path)



# python data_preprocessor.py --incoming_folder ../../data_processed/WiSec_unmonitored_trimmed_5_incoming --outgoing_folder ../../data_processed/WiSec_unmonitored_trimmed_5_outgoing --csv_file_path ../../data_processed/WiSec_unmonitored_trimmed_5_features.csv
    
# python data_preprocessor.py --incoming_folder ../../data_processed/Wi_incoming --outgoing_folder ../../data_processed/Wi_outgoing --csv_file_path ../../data_processed/Wi.csv

# python data_preprocessor.py --incoming_folder ../../data_processed/test/traffic_W_incoming --outgoing_folder ../../data_processed/test/traffic_W_outgoing --csv_file_path ../../data_processed/test/traffic_W.csv