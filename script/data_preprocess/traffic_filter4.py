import subprocess
import os
import argparse
import yaml
from concurrent.futures import ProcessPoolExecutor

def ensure_dir_exists(directory):
    """Ensure the specified directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def find_pcap_files(directory):
    """Recursively find all .pcap files within the given directory."""
    pcap_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.pcap'):
                pcap_files.append(os.path.join(root, file))
    return pcap_files

def load_alexa_ips_from_config(config_file):
    """Load Alexa IP addresses from the YAML configuration file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return {subfolder['subfolderName']: subfolder['EchoIP'] for subfolder in config['subfolders']}

def filter_traffic_task(pcap_file, echo_ip, base_outgoing_dir, base_incoming_dir, total_folder):
    """A task to filter and save the outgoing and incoming traffic for a single .pcap file."""
    relative_path = os.path.relpath(pcap_file, start=total_folder)
    base_name = os.path.splitext(os.path.basename(pcap_file))[0]
    
    outgoing_dir = os.path.join(base_outgoing_dir, os.path.dirname(relative_path))
    incoming_dir = os.path.join(base_incoming_dir, os.path.dirname(relative_path))
    
    ensure_dir_exists(outgoing_dir)
    ensure_dir_exists(incoming_dir)
    
    outgoing_pcap = os.path.join(outgoing_dir, f"{base_name}_outgoing.pcap")
    incoming_pcap = os.path.join(incoming_dir, f"{base_name}_incoming.pcap")



    outgoing_command = f"tshark -r \"{pcap_file}\" -Y \"ip.src == {echo_ip}\" -w \"{outgoing_pcap}\""
    incoming_command = f"tshark -r \"{pcap_file}\" -Y \"ip.dst == {echo_ip}\" -w \"{incoming_pcap}\""

    subprocess.run(outgoing_command, shell=True)
    subprocess.run(incoming_command, shell=True)

    print(f"Filtered outgoing traffic saved to {outgoing_pcap}")
    print(f"Filtered incoming traffic saved to {incoming_pcap}")

def main():
    parser = argparse.ArgumentParser(description='Filter network traffic based on Alexa IP configurations.')
    parser.add_argument('--total_folder', type=str, required=True, help='Top-level directory containing traffic data (.pcap files) in subdirectories')
    parser.add_argument('--base_outgoing_dir', type=str, required=True, help='Base directory to save filtered outgoing traffic data')
    parser.add_argument('--base_incoming_dir', type=str, required=True, help='Base directory to save filtered incoming traffic data')
    parser.add_argument('--config', type=str, required=True, help='YAML configuration file with Alexa IPs and subfolder names')
    args = parser.parse_args()

    alexa_ips = load_alexa_ips_from_config(args.config)
    pcap_files = find_pcap_files(args.total_folder)

    with ProcessPoolExecutor() as executor:
        for pcap_file in pcap_files:
            subfolder_name = os.path.basename(os.path.dirname(pcap_file))
            echo_ip = alexa_ips.get(subfolder_name)
            if echo_ip:
                executor.submit(filter_traffic_task, pcap_file, echo_ip, args.base_outgoing_dir, args.base_incoming_dir, args.total_folder)

if __name__ == "__main__":
    main()

# python traffic_filter4.py --total_folder ../../data_processed/WiSec_unmonitored_trimmed_5 --base_outgoing_dir ../../data_processed/WiSec_unmonitored_trimmed_outgoing --base_incoming_dir ../../data_processed/WiSec_unmonitored_5_trimmed_incoming --config ../../config/WiSec_unmonitored.yaml