import os
from concurrent.futures import ThreadPoolExecutor
from scapy.all import rdpcap, wrpcap
# import shutil

def trim_pcap(input_pcap_path, output_pcap_path, head_trim_sec=0.5, tail_trim_sec=1.0):
    packets = rdpcap(input_pcap_path)
    start_time = packets[0].time
    end_time = packets[-1].time
    trimmed_packets = [pkt for pkt in packets if (pkt.time >= start_time + head_trim_sec) and (pkt.time <= end_time - tail_trim_sec)]
    wrpcap(output_pcap_path, trimmed_packets)

def process_subfolder(root, subfolder, output_root, head_trim_sec, tail_trim_sec):
    input_folder = os.path.join(root, subfolder)
    output_folder = os.path.join(output_root, subfolder)
    os.makedirs(output_folder, exist_ok=True)
    
    pcap_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.pcap')], key=lambda x: int(x.split('.')[0]))
    
    for file in pcap_files:
        input_pcap_path = os.path.join(input_folder, file)
        output_pcap_path = os.path.join(output_folder, file)
        trim_pcap(input_pcap_path, output_pcap_path, head_trim_sec, tail_trim_sec)

def parallel_trim(root_folder, output_root_folder, head_trim_sec=0.5, tail_trim_sec=1.0):
    subfolders = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))], key=lambda x: int(x))
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_subfolder, root_folder, subfolder, output_root_folder, head_trim_sec, tail_trim_sec) for subfolder in subfolders]
        # Wait for all threads to complete
        for future in futures:
            future.result()

# Example usage:
parallel_trim('../../data_processed/vc_200/alexa/total3/', '../../data_processed/vc_200/alexa/trimmed_05_1/', head_trim_sec=0.5, tail_trim_sec=1.0)
