import subprocess
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

def ensure_dir_exists(directory):
    """Ensure the specified directory exists, create if not."""
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

def pcap_to_csv(pcap_path, csv_path):
    """Convert pcap file to csv format using tshark."""
    command = [
        "tshark",
        "-r", str(pcap_path),
        "-T", "fields",
        "-t", "ad",  # absolute date time format
        "-e", "frame.number",
        "-e", "frame.time",
        "-e", "ip.src",
        "-e", "ip.dst",
        "-e", "_ws.col.Protocol",
        "-e", "frame.len",
        "-E", "header=y",
        "-E", "separator=,",
        "-E", "quote=d",
        "-E", "occurrence=f"
    ]
    ensure_dir_exists(csv_path.parent)
    with open(csv_path, 'w') as output_file:
        subprocess.run(command, stdout=output_file, check=True, text=True)

def find_pcap_files(directory):
    """Recursively find all .pcap files within the given directory."""
    return [f for f in Path(directory).rglob('*.pcap')]

def convert_file(args):
    """Unpack arguments and call pcap_to_csv."""
    pcap_to_csv(*args)

def main():
    parser = argparse.ArgumentParser(description='Convert .pcap files to .csv format for analysis, preserving directory structure.')
    parser.add_argument('--pcap_dir', type=str, required=True, help='Directory containing .pcap files to process')
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory to save the resulting .csv files')
    args = parser.parse_args()

    pcap_dir = Path(args.pcap_dir)
    csv_dir = Path(args.csv_dir)

    pcap_files = find_pcap_files(pcap_dir)
    tasks = []
    for pcap_path in pcap_files:
        relative_path = pcap_path.relative_to(pcap_dir)
        csv_path = csv_dir / relative_path.with_suffix('.csv')
        tasks.append((pcap_path, csv_path))

    with ProcessPoolExecutor() as executor:
        executor.map(convert_file, tasks)

    print(f"Conversion complete. Check the {csv_dir} directory for the output files.")

if __name__ == "__main__":
    main()


# python pcap2csv2.py --pcap_dir "../data_processed/traffic_W_incoming" --csv_dir "../data_processed/traffic_W_incoming_csv"
# python pcap2csv2.py --pcap_dir "../data_processed/traffic_W_outgoing" --csv_dir "../data_processed/traffic_W_outgoing_csv"
