import logging
import os
import subprocess
import paramiko
import sounddevice as sd
import numpy as np
import argparse
import time
from datetime import datetime, timedelta
import platform

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

def play_audio(file_name):
    # Construct the absolute file path
    current_working_directory = os.getcwd()
    file_path = os.path.join(current_working_directory, file_name)
    
    # Log the absolute path to ensure it's correct
    logging.info(f"Absolute path to audio file: {file_path}")
    
    # Determine the OS and use the appropriate command to play audio
    if platform.system() == "Windows":
        subprocess.call(["start", "wmplayer", file_path], shell=True)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["afplay", file_path])
    elif platform.system() == "Linux":
        subprocess.call(["aplay", file_path])
    else:
        raise OSError("Unsupported operating system for audio playback")
    
    logging.info(f"Finished playing audio file: {file_path}")

def silence_detection(ssh, pcap_filename, silence_duration_threshold=3, volume_threshold=5, device_index=1):
    print("Starting silence detection using device index:", device_index)
    last_sound_time = datetime.now()
    should_exit = False

    def audio_callback(indata, frames, time, status):
        nonlocal last_sound_time, should_exit
        if should_exit: 
            return
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm > volume_threshold:
            last_sound_time = datetime.now()
        elif (datetime.now() - last_sound_time).seconds > silence_duration_threshold:
            silence_detected_time = last_sound_time + timedelta(seconds=silence_duration_threshold)
            print(f"Silence detected at {silence_detected_time}. Preparing to stop network traffic capture.")
            should_exit = True

    try:
        # Query the default input device for its supported channel count
        device_info = sd.query_devices(device_index, 'input')
        channel_count = device_info['max_input_channels']  # Use the max number of input channels
        print(f"Device supports {channel_count} input channels.")

        with sd.InputStream(callback=audio_callback, device=device_index, channels=channel_count):
            print("Listening on device {} with {} channels...".format(device_index, channel_count))
            while not should_exit:
                sd.sleep(1000)
    except Exception as e:
        print(f"An error occurred in silence_detection: {e}")
        ssh.execute_command("ps | grep '[t]cpdump' | awk '{print $1}' | xargs kill")
        raise e  # Raising the exception will stop the whole program

    try:
        # Stop tcpdump here, after exiting the InputStream context
        ssh.execute_command("ps | grep '[t]cpdump' | awk '{print $1}' | xargs kill")
        print("Network traffic capture stopped.")
    except Exception as e:
        print(f"Failed to stop tcpdump on remote device: {e}")

    print("Silence detection done")




def play_and_capture_traffic(ssh, audio_file, audio_dir, start_repetition, repetitions):
    audio_file_base_name = os.path.splitext(audio_file)[0]
    # pcap_dir = f"/opt/tcpdump/{audio_file_base_name}/"
    pcap_dir = f"/opt/simple_50_win/{audio_file_base_name}/"
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
        # print("Silence detection done")
        # silence_detection(ssh, pcap_filename, input_device_id=2)



def main(ip, username, password, audio_dir, repetitions, start_file_number, start_repetition):
    print("Starting main function")
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

    # main(args.ip, args.username, args.password, args.audio_dir, args.repetitions, args.start_file_number, args.start_repetition)
    print(sd.query_devices())


# python data_collector_win.py --ip 192.168.4.1 --username root --password raspberry --audio_dir audioPlay_A --start_file_number 2 --start_repetition 1
# python data_collector_win.py --ip 192.168.4.1 --username root --password raspberry --audio_dir audioPlay_A 

# print("Script execution started")