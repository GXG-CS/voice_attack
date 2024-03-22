import os
import csv
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import find_peaks
from numpy.fft import fft
from scapy.all import rdpcap
from glob import glob

def parse_pcap(file_path):
    packets = rdpcap(file_path)
    sizes = np.array([len(p) for p in packets]) if packets else np.array([])
    times = np.array([p.time for p in packets]) if packets else np.array([])
    inter_arrival_times = np.diff(times) if len(times) > 1 else np.array([])
    return sizes, inter_arrival_times

def calculate_features(sizes):
    if sizes.size == 0:
        return {
            'mean': np.nan, 'std_dev': np.nan, 'variance': np.nan,
            'max_value': np.nan, 'min_value': np.nan, 'range': np.nan,
            'median': np.nan, 'mad': np.nan, 'skewness': np.nan, 'kurtosis': np.nan,
            'entropy': np.nan
        }
    features = {
        'mean': np.mean(sizes), 'std_dev': np.std(sizes), 'variance': np.var(sizes),
        'max_value': np.max(sizes), 'min_value': np.min(sizes),
        'range': np.max(sizes) - np.min(sizes), 'median': np.median(sizes),
        'mad': np.mean(np.abs(sizes - np.mean(sizes))), 'skewness': skew(sizes),
        'kurtosis': kurtosis(sizes), 'entropy': entropy(np.histogram(sizes, bins=10)[0])
    }
    return features



def parse_pcap(file_path):
    packets = rdpcap(file_path)
    sizes = np.array([len(p) for p in packets]) if packets else np.array([])
    return sizes



def extract_features_from_pcap(file_path):
    sizes = parse_pcap(file_path)  # Updated to receive a single value.
    return calculate_features(sizes)


def process_file(file_path):
    features, error = extract_features_from_pcap(file_path), None
    return features, error

def process_folder(folder_path, label, executor):
    features_list = []
    file_paths = sorted(glob(os.path.join(folder_path, f'simple_{label}', '*.pcap')))
    futures = [executor.submit(process_file, file_path) for file_path in file_paths]
    for future in futures:
        features, error = future.result()
        if features and not error:
            features_list.append(features)
    return features_list

def get_labels(folder_path):
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    labels = [subfolder.split('_')[-1] for subfolder in subfolders]
    return sorted(set(labels), key=int)

def write_features_to_csv(combined_features, csv_file_path):
    with open(csv_file_path, 'w', newline='') as file:
        fieldnames = ['label'] + [f'{feat}_incoming' for feat in combined_features[0][1]] + [f'{feat}_outgoing' for feat in combined_features[0][1]]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for label, inc_features, out_features in combined_features:
            row = {'label': label}
            row.update({f'{k}_incoming': v for k, v in inc_features.items()})
            row.update({f'{k}_outgoing': v for k, v in out_features.items()})
            writer.writerow(row)

def main(incoming_folder, outgoing_folder, csv_file_path):
    labels = get_labels(incoming_folder)  # Assuming outgoing_folder has matching labels
    combined_features = []

    with ProcessPoolExecutor() as executor:
        for label in labels:
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