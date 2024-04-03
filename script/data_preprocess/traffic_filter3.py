import subprocess
import os
import argparse
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

def filter_traffic_task(pcap_file, alexa_ip, base_outgoing_dir, base_incoming_dir, total_folder, use_vpn_ip=None, vpn_ip=None):
    """A task to filter and save the outgoing and incoming traffic for a single .pcap file."""
    relative_path = os.path.relpath(pcap_file, start=total_folder)
    base_name = os.path.splitext(os.path.basename(pcap_file))[0]
    
    outgoing_dir = os.path.join(base_outgoing_dir, os.path.dirname(relative_path))
    incoming_dir = os.path.join(base_incoming_dir, os.path.dirname(relative_path))
    
    ensure_dir_exists(outgoing_dir)
    ensure_dir_exists(incoming_dir)
    
    outgoing_pcap = os.path.join(outgoing_dir, f"{base_name}_outgoing.pcap")
    incoming_pcap = os.path.join(incoming_dir, f"{base_name}_incoming.pcap")

    # Command for filtering outgoing packets
    if use_vpn_ip:
        outgoing_command = f"tshark -r \"{pcap_file}\" -Y \"ip.src == {alexa_ip} and ip.dst == {vpn_ip}\" -w \"{outgoing_pcap}\""
    else:
        outgoing_command = f"tshark -r \"{pcap_file}\" -Y \"ip.src == {alexa_ip}\" -w \"{outgoing_pcap}\""

    # Command for filtering incoming packets
    if use_vpn_ip:
        incoming_command = f"tshark -r \"{pcap_file}\" -Y \"ip.src == {vpn_ip} and ip.dst == {alexa_ip}\" -w \"{incoming_pcap}\""
    else:
        incoming_command = f"tshark -r \"{pcap_file}\" -Y \"ip.dst == {alexa_ip}\" -w \"{incoming_pcap}\""

    subprocess.run(outgoing_command, shell=True)
    subprocess.run(incoming_command, shell=True)

    return f"Filtered outgoing traffic saved to {outgoing_pcap}\nFiltered incoming traffic saved to {incoming_pcap}"

def main():
    parser = argparse.ArgumentParser(description='Filter network traffic between Alexa and VPN server into separate directories, preserving subdirectory structure.')
    parser.add_argument('--total_folder', type=str, required=True, help='Top-level directory containing traffic data (.pcap files) in subdirectories')
    parser.add_argument('--base_outgoing_dir', type=str, required=True, help='Base directory to save filtered outgoing traffic data, preserving subdirectory structure')
    parser.add_argument('--base_incoming_dir', type=str, required=True, help='Base directory to save filtered incoming traffic data, preserving subdirectory structure')
    parser.add_argument('--alexa_ip', type=str, required=True, help='IP address of the Alexa device')
    parser.add_argument('--vpn_ip', type=str, help='IP address of the VPN server')
    parser.add_argument('--use_vpn', action='store_true', help='Flag to indicate whether to use VPN IP in filtering')
    args = parser.parse_args()

    pcap_files = find_pcap_files(args.total_folder)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(filter_traffic_task, pcap_file, args.alexa_ip, args.base_outgoing_dir, args.base_incoming_dir, args.total_folder, args.use_vpn, args.vpn_ip) for pcap_file in pcap_files]
        for future in futures:
            print(future.result())

if __name__ == "__main__":
    main()


# python traffic_filter3.py --total_folder "../../raw_data/current_categories/traffic_W_raw" --base_outgoing_dir "../../data_processed/categories/categories_outgoing" --base_incoming_dir "../../data_processed/categories/categories_incoming" --alexa_ip "10.0.0.159" 
# 