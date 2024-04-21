import os
import argparse
from scapy.all import rdpcap, wrpcap
from concurrent.futures import ProcessPoolExecutor, as_completed

def remove_ds_store(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == '.DS_Store':
                os.remove(os.path.join(root, file))
                print(f"Removed .DS_Store from {root}")

def process_pcap(file_path, output_file_path, alexa_ip):
    packets = rdpcap(file_path)
    # total
    # filtered_packets = [pkt for pkt in packets if pkt.haslayer('IP') and alexa_ip in pkt['IP'].src]
    filtered_packets = [pkt for pkt in packets if pkt.haslayer('IP') and (pkt['IP'].src == alexa_ip or pkt['IP'].dst == alexa_ip)]
    
    # incoming
    # filtered_packets = [pkt for pkt in packets if pkt.haslayer('IP') and pkt['IP'].dst == alexa_ip]
    
    # outgoing
    # filtered_packets = [pkt for pkt in packets if pkt.haslayer('IP') and pkt['IP'].src == alexa_ip]
    wrpcap(output_file_path, filtered_packets)

def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def process_subdir(subdir, root_dir, output_root_dir, alexa_ip):
    subdir_path = os.path.join(root_dir, subdir)
    output_subdir_path = os.path.join(output_root_dir, subdir)
    if not os.path.exists(output_subdir_path):
        os.makedirs(output_subdir_path)
    for file_name in sorted(os.listdir(subdir_path), key=lambda x: int(x.split('.')[0])):
        if file_name.endswith('.pcap'):
            file_path = os.path.join(subdir_path, file_name)
            output_file_path = os.path.join(output_subdir_path, file_name)
            process_pcap(file_path, output_file_path, alexa_ip)
    return f"Completed processing for {subdir}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter PCAP files for a specific IP address.')
    parser.add_argument('--root_dir', type=str, required=True, help='Path to the directory containing traffic subfolders')
    parser.add_argument('--output_root_dir', type=str, required=True, help='Path to the output directory where filtered PCAP files will be stored')
    parser.add_argument('--alexa_ip', type=str, required=True, help='IP address to filter the traffic by')
    args = parser.parse_args()

    remove_ds_store(args.root_dir)
    remove_ds_store(args.output_root_dir)

    if not os.path.exists(args.output_root_dir):
        os.makedirs(args.output_root_dir)

    with ProcessPoolExecutor(max_workers=None) as executor:  # None defaults to the number of processors on the machine
        futures = [executor.submit(process_subdir, subdir, args.root_dir, args.output_root_dir, args.alexa_ip)
                   for subdir in sorted(filter(is_int, os.listdir(args.root_dir)), key=int)]
        for future in as_completed(futures):
            print(future.result())


# python traffic_filter.py --root_dir "../../raw_data/vc_200/alexa/traffic" --output_root_dir "../../raw_data/vc_200/alexa/total3" --alexa_ip "10.0.0.159"
# python traffic_filter.py --root_dir "../../vc_200_alexa" --output_root_dir "../../data_processed/vc_200/alexa/total3" --alexa_ip "10.0.0.159"
# python traffic_filter.py --root_dir "../../data_processed/vc_200/alexa/total3" --output_root_dir "../../data_processed/vc_200/alexa/incoming" --alexa_ip "10.0.0.159"
# python traffic_filter.py --root_dir "../../data_processed/vc_200/alexa/total3" --output_root_dir "../../data_processed/vc_200/alexa/outgoing" --alexa_ip "10.0.0.159"


            