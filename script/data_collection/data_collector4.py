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

def play_audio(file):
    logging.info(f"Starting to play audio file: {file}")
    subprocess.call(["afplay", file])  # Adjust command based on the OS
    logging.info(f"Finished playing audio file: {file}")

def combined_capture_control(ssh, audio_file, audio_dir, pcap_filename, max_duration=60, stop_audio='stop_signal_audio.wav', silence_threshold=4):
    volume_threshold = 4
    start_time = datetime.now()
    last_sound_time = start_time
    should_exit = False

    def audio_callback(indata, frames, time, status):
        nonlocal last_sound_time, should_exit
        current_time = datetime.now()
        if np.linalg.norm(indata) * 10 > volume_threshold:
            last_sound_time = current_time
        # Check for silence duration exceeded
        if (current_time - last_sound_time).seconds > silence_threshold:
            # logging.info(f"Silence detected for {silence_threshold} seconds. Stopping capture.")
            should_exit = True
        # Check for total duration exceeded
        if (current_time - start_time).seconds >= max_duration:
            logging.info(f"Maximum duration of {max_duration} seconds reached. Stopping capture.")
            play_audio(stop_audio)  # Play stop signal audio if max duration is reached
            should_exit = True

    with sd.InputStream(callback=audio_callback):
        # ssh.execute_command(f"nohup tcpdump -i eth0 -w {pcap_filename} > /dev/null 2>&1 &")
        while not should_exit:
            time.sleep(1)
        time.sleep(1)

    ssh.execute_command("ps | grep '[t]cpdump' | awk '{print $1}' | xargs -r kill -SIGINT")
    logging.info("Network traffic capture stopped.")

def play_and_capture_traffic(ssh, audio_file, audio_dir, repetition):
    logging.info(f"Starting to play and capture for {audio_file}, repetition {repetition}")
    audio_file_base_name = os.path.splitext(audio_file)[0]
    pcap_dir = f"/opt/vc_200_alexa/{audio_file_base_name}/"
    ssh.execute_command(f"mkdir -p {pcap_dir}")
    ssh.execute_command(f"mount /dev/mmcblk0p3 /opt")

    pcap_filename = f"{pcap_dir}{repetition}.pcap"
    
    ssh.execute_command(f"nohup tcpdump -i eth0 -w {pcap_filename} > /dev/null 2>&1 &")

    play_audio(os.path.join(audio_dir, audio_file))
    combined_capture_control(ssh, audio_file, audio_dir, pcap_filename)
    logging.info(f"Finished playing and capturing for {audio_file}, repetition {repetition}")

def main(ip, username, password, audio_dir, repetitions):
    logging.info("Starting main function")
    with SSHConnection(ip, username, password) as ssh:
        for repetition in range(201, repetitions + 50):
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]
            audio_files = sorted(audio_files, key=lambda f: int(os.path.splitext(f)[0]))
            
            for audio_file in audio_files:
                play_and_capture_traffic(ssh, audio_file, audio_dir, repetition)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play audio files and capture network traffic with silence and duration controls.')
    parser.add_argument('--ip', required=True, help='IP address of the device')
    parser.add_argument('--username', required=True, help='Username for SSH')
    parser.add_argument('--password', required=True, help='Password for SSH')
    parser.add_argument('--audio_dir', required=True, help='Directory with audio files')
    parser.add_argument('--repetitions', type=int, default=200, help='Number of repetitions for the process')
    args = parser.parse_args()

    main(args.ip, args.username, args.password, args.audio_dir, args.repetitions)
