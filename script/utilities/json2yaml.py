import json
import yaml

# Load the JSON data
with open(r'C:\Users\GXG\Research\voice_attack\raw_data\dilawer11\simple_100_alexa\simple_100_alexa.json', 'r') as json_file:
    commands = json.load(json_file)

# Initialize the YAML structure
yaml_data = {"subfolders": []}

# Fixed Echo IP
echo_ip = "10.0.0.159"

# Convert each command into the required YAML format
for i, command in enumerate(commands, start=1):
    subfolder_name = f"simple_{i}"
    voice_command = command["invokePhrase"]
    yaml_data["subfolders"].append({
        "subfolderName": subfolder_name,
        "voiceCommand": voice_command,
        "EchoIP": echo_ip
    })

# Save the converted data to a YAML file
with open('commands.yaml', 'w') as yaml_file:
    yaml.dump(yaml_data, yaml_file, default_flow_style=False, sort_keys=False)

print("Conversion completed. Check the 'commands.yaml' file.")
