import logging
import os
import subprocess
import paramiko
import sounddevice as sd
import numpy as np
import argparse
import time
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(filename='data_collection.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

class SSHConnection:
    def __init__(self, ip, username, password=None):
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if password:
            self.ssh.connect(ip, username=username, password=password)
        else:
            raise ValueError("Password is required for SSH connection")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.ssh.close()

    def execute_command(self, command):
        stdin, stdout, stderr = self.ssh.exec_command(command)
        return stdout.read().decode(), stderr.read().decode()

def get_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def play_audio(file):
    logging.info(f"Starting to play audio file: {file}")
    subprocess.call(["afplay", file])  # Adjust command based on your OS
    logging.info(f"Finished playing audio file: {file}")

def silence_detection(ssh, pcap_filename, silence_duration_threshold=3):
    volume_threshold = 5
    last_sound_time = datetime.now()
    should_exit = False

    def audio_callback(indata, frames, time, status):
        nonlocal last_sound_time, should_exit
        if should_exit:  # Check if we should exit and return immediately if true
            return
        current_time = datetime.now()
        if np.linalg.norm(indata) * 10 > volume_threshold:
            last_sound_time = current_time
        elif (current_time - last_sound_time).seconds > silence_duration_threshold:
            silence_detected_time = last_sound_time + timedelta(seconds=silence_duration_threshold)
            logging.info(f"Silence detected at {silence_detected_time}. Preparing to stop network traffic capture.")
            should_exit = True

    with sd.InputStream(callback=audio_callback):
        while not should_exit:
            pass  # Just wait until silence is detected
        time.sleep(1)  # Sleep briefly to ensure all packets are captured

    # Stop tcpdump here, after exiting the InputStream context
    ssh.execute_command("ps | grep '[t]cpdump' | awk '{print $1}' | xargs -r kill -SIGINT")
    logging.info("Network traffic capture stopped.")




def play_and_capture_traffic(ssh, audio_file, audio_dir, start_repetition, repetitions):
    audio_file_base_name = os.path.splitext(audio_file)[0]
    pcap_dir = f"/opt/tcpdump/{audio_file_base_name}/"
    ssh.execute_command(f"mkdir -p {pcap_dir}")
    ssh.execute_command(f"mount /dev/mmcblk0p3 /opt")


    for i in range(start_repetition - 1, repetitions):
        logging.info(f"Processing {i+1}/{repetitions} for {audio_file}")
        full_audio_path = os.path.join(audio_dir, audio_file)
        pcap_filename = f"{pcap_dir}{i+1}.pcap"

        # Start network traffic capture before playing audio
        logging.info(f"Starting network traffic capture. Captured data will be saved in: {pcap_filename}")
        # ssh.execute_command(f"nohup tcpdump -i eth0 -w {pcap_filename} > /dev/null 2>&1 &")
        ssh.execute_command(f"nohup tcpdump -i eth0 -w {pcap_filename} > /dev/null 2>&1 &")

        # Play audio immediately after starting capture
        play_audio(full_audio_path)
        
        # Wait for silence detection to stop the capture
        silence_detection(ssh, pcap_filename)


def main(ip, username, password, audio_dir, repetitions, start_file_number, start_repetition):
    logging.info("Starting main function")
    with SSHConnection(ip, username, password) as ssh:
        audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
        if start_file_number > 0 and start_file_number <= len(audio_files):
            start_index = start_file_number - 1
        else:
            start_index = 0
        for audio_file in audio_files[start_index:]:
            play_and_capture_traffic(ssh, audio_file, audio_dir, start_repetition, repetitions)
            start_repetition = 1  # Reset to 1 for subsequent files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play audio files in alphabetical order and capture network traffic based on silence detection.')
    parser.add_argument('--ip', required=True, help='IP address of the Raspberry Pi')
    parser.add_argument('--username', required=True, help='Username for SSH login')
    parser.add_argument('--password', required=True, help='Password for SSH login')
    parser.add_argument('--audio_dir', required=True, help='Directory containing audio files to play')
    parser.add_argument('--repetitions', type=int, default=1000, help='Total number of times to repeat process for each audio file')
    parser.add_argument('--start_file_number', type=int, default=1, help='The ordinal number of the file to start playback from')
    parser.add_argument('--start_repetition', type=int, default=1, help='The repetition number to start from for the specified file')
    args = parser.parse_args()

    main(args.ip, args.username, args.password, args.audio_dir, args.repetitions, args.start_file_number, args.start_repetition)

# python data_collector.py --ip 192.168.4.1 --username root --password raspberry --audio_dir audioPlay_A --start_file_number 2 --start_repetition 1
# python data_collector.py --ip 192.168.4.1 --username root --password raspberry --audio_dir audioPlay_A 