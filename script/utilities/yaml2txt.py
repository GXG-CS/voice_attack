import os
import yaml

# Define the path to your YAML file and the folder where you want to save the text files
yaml_file_path = 'C:\\Users\\GXG\\Research\\voice_attack\\config\\WiSec_unmonitor.yaml'
output_folder = 'C:\\Users\\GXG\\Research\\voice_attack\\raw_data\\WiSec\\unmonitor_100\\text_A'

# Ensure the output folder exists, create it if not
os.makedirs(output_folder, exist_ok=True)

# Load the YAML file
with open(yaml_file_path, 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# Extract voice commands from the loaded YAML data
voice_commands = [item['voiceCommand'] for item in data['subfolders']]

# Create and write each command to a separate file in the specified folder, prefixed with "Alexa,"
for i, command in enumerate(voice_commands, start=1):
    file_path = os.path.join(output_folder, f"{i}.txt")
    with open(file_path, 'w') as file:
        file.write(f"Alexa, {command}\n")

print("Files have been created in the specified folder.")
