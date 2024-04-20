import logging
import os
import subprocess
import paramiko
import sounddevice as sd
import numpy as np
import argparse
import time
from datetime import datetime, timedelta

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
    subprocess.call(["afplay", file])  # Adjust command based on the OS
    logging.info(f"Finished playing audio file: {file}")

def silence_detection(ssh, pcap_filename, silence_duration_threshold=3):
    volume_threshold = 5
    last_sound_time = datetime.now()
    should_exit = False

    def audio_callback(indata, frames, time, status):
        nonlocal last_sound_time, should_exit
        if should_exit:
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
            pass
        time.sleep(1)

    ssh.execute_command("ps | grep '[t]cpdump' | awk '{print $1}' | xargs -r kill -SIGINT")
    logging.info("Network traffic capture stopped.")

def play_and_capture_traffic(ssh, audio_file, audio_dir, repetition):
    # Modified to include repetition in the pcap filename
    logging.info(f"Starting to play and capture for {audio_file}, repetition {repetition}")
    audio_file_base_name = os.path.splitext(audio_file)[0]
    pcap_dir = f"/opt/tcpdump/{audio_file_base_name}/"
    ssh.execute_command(f"mkdir -p {pcap_dir}")
    ssh.execute_command(f"mount /dev/mmcblk0p3 /opt")

    pcap_filename = f"{pcap_dir}{repetition}.pcap"  # Naming pcap file according to repetition number
    # Start capturing packets
    ssh.execute_command(f"nohup tcpdump -i eth0 -w {pcap_filename} > /dev/null 2>&1 &")
    play_audio(os.path.join(audio_dir, audio_file))
    silence_detection(ssh, pcap_filename)
    logging.info(f"Finished playing and capturing for {audio_file}, repetition {repetition}")

def main(ip, username, password, audio_dir, repetitions):
    # Main logic to handle repetitions and iterate over audio files
    logging.info("Starting main function")
    with SSHConnection(ip, username, password) as ssh:
        for repetition in range(50, repetitions + 1):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            # Sort files numerically based on the number in the filename
            audio_files = sorted(audio_files, key=lambda f: int(os.path.splitext(f)[0]))
            
            for audio_file in audio_files:
                play_and_capture_traffic(ssh, audio_file, audio_dir, repetition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play audio files and capture network traffic in iterations.')
    parser.add_argument('--ip', required=True, help='IP address of the device')
    parser.add_argument('--username', required=True, help='Username for SSH')
    parser.add_argument('--password', required=True, help='Password for SSH')
    parser.add_argument('--audio_dir', required=True, help='Directory with audio files')
    parser.add_argument('--repetitions', type=int, default=200, help='Number of repetitions for the process')
    args = parser.parse_args()

    main(args.ip, args.username, args.password, args.audio_dir, args.repetitions)

# python data_collector.py --ip 192.168.4.1 --username root --password raspberry --audio_dir audioPlay_A --start_file_number 2 --start_repetition 1
# python data_collector.py --ip 192.168.4.1 --username root --password raspberry --audio_dir audioPlay_A 