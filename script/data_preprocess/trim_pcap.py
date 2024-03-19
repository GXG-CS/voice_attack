from scapy.all import rdpcap, wrpcap
import os

def trim_pcap(input_file_path, output_file_path, trim_duration=0.5):
    """
    Trims the first 'trim_duration' seconds from a pcap file and saves the result.
    
    Parameters:
    - input_file_path: Path to the input .pcap file.
    - output_file_path: Path where the trimmed .pcap file will be saved.
    - trim_duration: Duration in seconds to trim from the start of the pcap file.
    """
    # Read the pcap file
    packets = rdpcap(input_file_path)
    
    if not packets:
        print("No packets found in the file.")
        return
    
    # Get the timestamp of the first packet
    start_time = packets[0].time
    
    # Identify the cutoff timestamp
    cutoff_time = start_time + trim_duration
    
    # Filter out packets occurring before the cutoff time
    trimmed_packets = [packet for packet in packets if packet.time >= cutoff_time]
    
    if not trimmed_packets:
        print("Trimming duration is longer than the pcap duration. No packets left.")
        return
    
    # Write the trimmed packets to a new file
    wrpcap(output_file_path, trimmed_packets)
    print(f"Trimmed pcap file saved to {output_file_path}")

# Example usage
input_file_path = '../../data_processed/traffic_W_outgoing/1/2_outgoing.pcap'
output_file_path = 'test.pcap'
trim_duration = 0.5  # or 1 for trimming the first 1 second

trim_pcap(input_file_path, output_file_path, trim_duration)
