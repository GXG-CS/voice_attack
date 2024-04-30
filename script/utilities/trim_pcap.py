import os
import logging
from scapy.all import rdpcap, wrpcap
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Set logging level for Scapy to suppress unnecessary warnings
logging.getLogger("scapy.runtime").setLevel(logging.ERROR)

def trim_pcap(input_file_path, output_file_path, trim_duration):
    """
    Trims the first 'trim_duration' seconds from a pcap file and saves the result.
    """
    try:
        packets = rdpcap(input_file_path)
        
        if not packets:
            print(f"No packets found in {input_file_path}.")
            return
        
        start_time = packets[0].time
        cutoff_time = start_time + trim_duration
        trimmed_packets = [packet for packet in packets if packet.time >= cutoff_time]
        
        if not trimmed_packets:
            print(f"Trimming duration is longer than the pcap duration in {input_file_path}. No packets left.")
            return
        
        wrpcap(output_file_path, trimmed_packets)
        print(f"Trimmed pcap file saved to {output_file_path}")
    except Exception as e:
        print(f"Exception while processing {input_file_path}: {e}")

def process_file(task):
    input_file_path, output_folder_path, trim_duration, input_base_path = task
    try:
        relative_path = os.path.relpath(os.path.dirname(input_file_path), input_base_path)
        output_directory = os.path.join(output_folder_path, relative_path)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        output_file_path = os.path.join(output_directory, os.path.basename(input_file_path))
        trim_pcap(input_file_path, output_file_path, trim_duration)
    except Exception as e:
        print(f"Error processing file {input_file_path}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim pcap files in a folder structure using parallel processing.")
    parser.add_argument("input_folder_path", help="Path to the main input folder containing .pcap files.")
    parser.add_argument("output_folder_path", help="Path to the main output folder where trimmed files will be saved.")
    parser.add_argument("--trim_duration", type=float, default=0.5, help="Duration in seconds to trim from the start of each pcap file.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder_path):
        os.makedirs(args.output_folder_path, exist_ok=True)

    tasks = []
    for root, dirs, files in os.walk(args.input_folder_path):
        for file in files:
            if file.endswith(".pcap"):
                input_file_path = os.path.join(root, file)
                tasks.append((input_file_path, args.output_folder_path, args.trim_duration, args.input_folder_path))
    
    # Limit the number of processes to avoid overloading
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_file, tasks)


# python trim_pcap.py ../../raw_data/WiSec/unmonitored_100/WiSec_unmonitored ../../data_processed/WiSec_unmonitored_trimmed --trim_duration 1