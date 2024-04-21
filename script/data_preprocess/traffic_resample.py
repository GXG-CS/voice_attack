import os
import pandas as pd
from scapy.all import rdpcap
from concurrent.futures import ThreadPoolExecutor

def resample_pcap(input_pcap_path, output_csv_path, resample_interval='200L'):
    packets = rdpcap(input_pcap_path)
    times = [pkt.time for pkt in packets]
    sizes = [len(pkt) for pkt in packets]
    times = [t - times[0] for t in times]
    
    df = pd.DataFrame({'Time': times, 'Size': sizes})
    df['Time'] = pd.to_timedelta(df['Time'].astype(float), unit='s')
    df.set_index('Time', inplace=True)
    
    resampled_df = df.resample(resample_interval).sum()
    resampled_df.reset_index(inplace=True)
    resampled_df['Time'] = resampled_df['Time'].dt.total_seconds()
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    resampled_df.to_csv(output_csv_path, index=False)

def process_subfolder(root, subfolder, output_root, resample_interval):
    input_folder = os.path.join(root, subfolder)
    output_folder = os.path.join(output_root, subfolder)
    pcap_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.pcap')], key=lambda x: int(x.split('.')[0]))
    
    for file_name in pcap_files:
        input_pcap_path = os.path.join(input_folder, file_name)
        output_csv_path = os.path.join(output_folder, file_name.replace('.pcap', '.csv'))
        resample_pcap(input_pcap_path, output_csv_path, resample_interval)

def parallel_resample(root_folder, output_root_folder, resample_interval='200L'):
    subfolders = sorted([f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))], key=lambda x: int(x))
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_subfolder, root_folder, subfolder, output_root_folder, resample_interval) for subfolder in subfolders]
        for future in futures:
            future.result()

# Example usage:
parallel_resample('../../data_processed/vc_200/alexa/total3/', '../../data_processed/vc_200/alexa/resampled_02s')
